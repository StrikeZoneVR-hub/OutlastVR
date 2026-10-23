// ol_xr.cpp - Outlast FlatXR: D3D9 -> (CPU readback) -> D3D11 -> OpenXR.
// =============================================================================
// Stage 2 of the Outlast lane: get the game's frame onto the headset, mono/flat.
// No camera work here - that's Stage 3+ (engine-level same-frame split).
//
// Per D3D9 frame, all on the game's Present thread (which owns the D3D11 context created here):
//   D3D9 GetBackBuffer -> CreateOffscreenPlainSurface(SYSTEMMEM) -> GetRenderTargetData
//   -> LockRect -> UpdateSubresource into a D3D11 BGRA texture (a locally created device)
//   -> xrAcquire/Wait -> CopyResource into the XR swapchain image -> xrRelease
//   -> submit ONE layer: head-locked quad (default) or a projection layer (P).
//
// Ported from Games/Dishonored/probe/DishonoredFlatXR/dish_xr.cpp, stripped to the
// parts that are known-good, and rebuilt x64. Outlast's Win64 OLGame.exe is x64, so
// none of the x86 OpenXR ABI corrections matter here - but the header is shared-safe.
//
// OpenXR is loaded dynamically (x64 openxr_loader.dll next to the game exe); no SDK link.
// =============================================================================

#include "ol_xr_types.h"
#include "ol_xr.h"
#include <d3d9.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3d9on12.h>
#include <dxgi.h>
#include <wincodec.h>
#include <cstdio>
#include <cstdarg>
#include <vector>
#include <math.h>

using namespace OLXR;

extern "C" IDirect3DSurface9* OLProxy_AcquireNativeStereo();

static void XLOG(const char* fmt, ...) {
    char buf[512]; va_list ap; va_start(ap, fmt);
    _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap); va_end(ap);
    OLProxyLog(buf);
}

// ---- state ----
static bool g_dead = false;      // init failed hard -> never retry
static bool g_inited = false;
static bool g_running = false;   // between xrBeginSession and STOPPING

static HMODULE g_loaderDll = nullptr;
static PFN_xrGetInstanceProcAddr g_getProc = nullptr;
static XrInstance  g_instance = 0;
static XrSystemId  g_system = 0;
static XrSession   g_session = 0;
static XrSwapchain g_swapchain = 0;
static XrSpace     g_viewSpace = 0;    // head-locked (quad layer)
static XrSpace     g_localSpace = 0;   // fixed world space (pose sampling)
static XrSpace     g_appSpace = 0;     // world-anchored space for the projection layer
static Functions   g_fn;

static ID3D11Device*        g_d3d11 = nullptr;
static ID3D11DeviceContext* g_ctx = nullptr;
static ID3D11Texture2D*     g_uploadTex = nullptr;
static std::vector<ID3D11Texture2D*> g_xrImages;
// ---- per-eye (SBS) path ----
// When the engine-side split is running, the backbuffer holds two half-width views. Each half
// goes to its own eye swapchain, and the layer is a real projection layer with the FOV the
// engine actually rendered.
static XrSwapchain g_scL = 0, g_scR = 0;
static std::vector<ID3D11Texture2D*> g_imgL, g_imgR;
// Source eye dimensions and headset submission dimensions are deliberately separate.
// Outlast's D3D9 renderer may be desktop-limited (for example 960x1080 per SBS half), while
// the OpenXR runtime normally wants a much larger image. Older builds made the swapchain equal
// to the source half, forcing the compositor to perform a very soft upscale. We now do the
// upscale ourselves into the runtime-recommended swapchain and apply a controlled sharpen.
static UINT g_srcEyeW = 0, g_srcEyeH = 0;
static UINT g_eyeW = 0, g_eyeH = 0;
static UINT g_runtimeEyeW = 0, g_runtimeEyeH = 0;
extern "C" int OLUE3_SetHiResEye(unsigned eyeW, unsigned eyeH);   // [HIRES] ol_ue3.cpp
static bool g_nativeEyeResolution = true;
static float g_upscaleSharpness = 0.35f;
// [UPSCALE] 1 = bicubic + contrast-adaptive sharpening (Test34 default), 0 = the Test28 bilinear +
// unsharp path exactly. [XR] Upscaler; both values live-switchable from the menu.
static volatile long g_upscaler = 1;
extern "C" int   OLXR_GetUpscaler() { return g_upscaler ? 1 : 0; }
extern "C" void  OLXR_SetUpscaler(int v) { g_upscaler = v ? 1 : 0; }
extern "C" float OLXR_GetSharpness() { return g_upscaleSharpness; }
extern "C" void  OLXR_SetSharpness(float v) { g_upscaleSharpness = !(v >= 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v); }
static bool g_eyeInitTried = false;
// AER holds the eye NOT rendered this frame. A swapchain hands back a rotating image index,
// so the previous frame's contents cannot be relied on - a separate copy per eye is kept and
// re-blit BOTH into their swapchains every frame. Sized to the source eye, so a mode change
// invalidates these too.
static ID3D11Texture2D* g_eyeHold[2] = { nullptr, nullptr };

static IDirect3DSurface9*   g_sysSurf = nullptr;   // D3D9 SYSTEMMEM readback surface
static UINT g_w = 0, g_h = 0;
static long long g_frame = 0;

// ---- head pose (exported; nothing consumes it yet) ----
volatile float g_olYawRad = 0.0f, g_olPitchRad = 0.0f, g_olRollRad = 0.0f;
// (Beta 1.1) [BODYTURN] Miles turned to face where you face: the recenter frame turns with him by exactly
// that much at the next locate, so the view stays put while his body (and your arms' frame) comes round.
// Right-positive millidegrees, posted from the game thread.
static volatile LONG g_bodyTurnPendMilli = 0;
extern "C" void OLXR_TurnBaseline(float rightDeg) { InterlockedExchangeAdd(&g_bodyTurnPendMilli, (LONG)lroundf(rightDeg * 1000.0f)); }
volatile float g_olHeadX = 0.0f, g_olHeadY = 0.0f, g_olHeadZ = 0.0f;
volatile long  g_olPoseValid = 0;
volatile float g_olDeclFovXDeg = 0.0f, g_olDeclFovYDeg = 0.0f;
volatile long  g_olUseProjection = 0;

static XrQuaternionf g_baseQ = { 0, 0, 0, 1 };
static XrVector3f    g_baseP = { 0, 0, 0 };
static XrPosef       g_headPoseApp;
static volatile long g_recenterReq = 1;      // recenter on the first valid pose
static volatile long g_recenterGeneration = 0; // lets body-relative IK re-zero with the XR frame
static volatile long g_centered = 0;

// [HEADPOS] POSITION BASELINE DISCIPLINE. The boot recenter fires on the first valid pose,
// while the game is still loading and the headset is usually on the desk. A yaw baseline
// taken there is harmless (one G press fixes it). A POSITION baseline taken there is not: every
// later head position is measured from desk height, the camera sits about a metre above the
// character and looks down at his own head. That is why lean/duck was removed from the menu.
// So position is only APPLIED once the baseline is trusted ("armed"):
//   - a manual recenter (key / pad / menu button) arms it on the spot - the player is
//     wearing the headset and facing forward by definition;
//   - otherwise, arm automatically ONCE: the headset must first MOVE well away from the boot
//     baseline (picked up off the desk), then hold still for a moment (on a head, reading the
//     menu). Position zero is taken at that still pose. A desk is dead still from the start
//     and never crosses the pickup threshold, so it can never arm; a head already wearing the
//     headset at launch never crosses it either and simply waits for G, as before.
// Rotation is untouched by all of this: yaw/pitch keep the shipped behaviour exactly.
static volatile long g_recenterManual = 0;   // the pending recenter was asked for by the player
static volatile long g_posArmed = 0;         // position baseline trusted: lean/duck may apply
static long          g_posAutoDone = 0;      // the automatic arm happens once per session
static long          g_posPickedUp = 0;      // moved > kPickupM from the boot baseline
static XrVector3f    g_settleRef = { 0, 0, 0 };
static XrTime        g_settleSince = 0;
static const float   kPickupM = 0.25f;       // desk -> head is 0.5-1.0 m; a lean is less
static const float   kStillM  = 0.02f;       // "holding still" radius
static const XrTime  kStillNs = 1000000000;  // for one second
extern "C" int OLXR_PosArmed() { return g_posArmed ? 1 : 0; }

// ---- menu-tunable stereo/tracking knobs -------------------------------------------------------
// CONVERGENCE slides the two eye images horizontally at SUBMIT, which moves the depth plane that
// sits at screen depth without touching the render. It is done here rather than by shearing the
// projection because shearing breaks fusion - a known dead end on this project.
// SMOOTHING is an exponential filter on the head angles; 0 keeps the raw, lowest-latency reading.
// CONVERGENCE, in ME1's units (2026-07-23). The slider is an "inward shift" in -3..+3 and ONE
// UNIT = 8% of the eye's width (ME1 sbs_splitter.cpp: conv = shift * 0.08). The old knob was a
// fixed +/-48 px - 2.6% of an 1832 px eye - so the ENTIRE Outlast range was smaller than ME1's
// DEFAULT (0.5 -> 4%), which is exactly why the slider read as "convergence doesn't work".
// Pixels are derived from the fraction at submit, so the feel survives resolution changes.
static const float kConvUnit = 0.08f;      // fraction of eye width per slider unit - ME1's number
static volatile float g_convShift = 0.5f;  // ME1's default inward shift

// ---- [STEREOMODEL] two complete stereo models, switchable live ---------------------------------
// Built 2026-08-04 to settle by experiment which model this game wants, rather than by argument.
// The two differ in exactly two behaviours and nothing else; everything upstream of here (the
// split, the camera offsets, head tracking) is shared.
//
// MODEL 0, "Outlast" - what this build has always shipped:
//   CONVERGENCE reserves SLACK. Each eye image is cut narrower by the shift amount so a plain box
//   copy can slide without running off the edge, and the declared frustum is narrowed to match.
//   Honest, but raising convergence visibly eats the left and right edges of your view.
//   SWAP flips the CAMERA only (ol_ue3), leaving the routing alone, so it genuinely inverts the
//   stereo: near objects get divergent disparity and cannot be fused. It is a rescue switch for a
//   backwards setup, not a quality control.
//
// MODEL 1, "Mass Effect" - how ME1/ME2 do it:
//   CONVERGENCE samples a SHIFTED window through a clamping sampler, so nothing is reserved and
//   no width is lost at any setting; the declared frustum stays the full symmetric half and the
//   small angular lie IS the toe-in. Sign is bolted to the HALF, not the eye.
//   SWAP flips the camera AND the routing, which cancels out for depth (left eye still sees the
//   left viewpoint, so no inversion and no near-object doubling) and instead INVERTS the
//   convergence plane, because the convergence sign rides the half. That inversion is the whole
//   visible effect of ME's swap toggle.
// DEFAULT IS 1, the Mass Effect model (chosen 2026-08-04 after a live A/B). Model 0 is kept
// because it is the whole history of this lane and reverting to it is one ini key, but it is no
// longer reachable from the menu: it costs field of view on every convergence change and its
// Swap inverts the stereo, and neither is something to hand a player.
static volatile long g_meModel = 1;        // 0 = Outlast, 1 = Mass Effect
extern "C" int  OLXR_GetStereoModel() { return g_meModel ? 1 : 0; }
extern "C" void OLXR_SetStereoModel(int v) { InterlockedExchange(&g_meModel, v ? 1 : 0); }
static int g_cropLx = 0, g_cropRx = 0;     // where each eye's window actually landed this frame
static float g_rtFovX = 0.0f, g_rtFovY = 0.0f;   // the HEADSET's own FOV, as reported by the runtime

// ---- [XRRUNTIME] runtime quirks, ported from ME1 (2026-08-04) -----------------------------------
// Two PC runtimes do not treat a declared projection FOV the way Virtual Desktop does, and both
// failures were found the hard way in the Mass Effect lane rather than reasoned about:
//  * META (Quest Link / Air Link, runtime name "Oculus" or "Meta") composites a projection layer
//    using ITS OWN per-eye asymmetric frustum from xrLocateViews and ignores a declared FOV that
//    differs. This mod's is symmetric, so each eye lands several degrees off in OPPOSITE directions:
//    permanent, unfusable doubling in every mode. VDXR honours the declared FOV, which is exactly
//    why this build has always fused here and would not have on a Link cable.
//  * STEAMVR shows NOTHING at all for a projection view declared WIDER than its own per-eye
//    frustum. endFrame succeeds, the session reports focused, the compositor stays void.
// One cure for both: declare the INTERSECTION of the declared window and the runtime's frustum,
// and crop the submitted rect to exactly that intersection, in tan space. No extra rendering, and
// "declared FOV must match what was rendered" stays true because the rect shrinks with it.
// Where the declared window is NARROWER than the runtime's, the intersection is that window and
// it no-ops.
static bool g_isOculusRuntime = false;
static bool g_isSteamVrRuntime = false;
static XrFovf g_rtEyeFov[2] = {};          // the runtime's own per-eye frustum, from xrLocateViews
static bool   g_rtEyeFovValid = false;
static XrPosef g_rtEyePose[2] = {};        // [XRSUBMIT] the runtime's own per-eye viewpoints
static float   g_rtEyeSep = 0.0f;          // metres between them, measured from xrLocateViews
extern "C" float OLXR_GetRtEyeSep() { return g_rtEyeSep; }
// [XRSUBMIT] the last declaration actually handed to the compositor, for the periodic dump.
static XrFovf  g_dbgDeclFov[2] = {};
static int32_t g_dbgRect[2][4] = {};       // x, y, w, h per eye
static XrPosef g_dbgDeclPose = {};
// [STEAMVRSCENE] a 64px opaque-black swapchain, cleared once, submitted as a placeholder
// PROJECTION layer under quad-only frames so SteamVR's waiting room does not take the screen.
static const int32_t kBlackSceneSize = 64;
// [SNAPTURN] While the game turns Miles for a snap turn, the headset shows black instead of the
// world sweeping past. ol_motion refreshes this every poll of the turn, so it ends ~one refresh
// after the turn does. 64-bit GetTickCount64 deadline (a 32-bit signed compare against an initial
// 0 reads as "active" for half of every 49.7-day uptime window); 0 = off.
static volatile ULONGLONG g_blinkUntilMs = 0;
extern "C" void OLXR_Blink(int ms) { g_blinkUntilMs = GetTickCount64() + (ULONGLONG)(ms > 0 ? ms : 0); }
static XrSwapchain g_blackSc = 0;
static std::vector<ID3D11Texture2D*> g_blackImgs;
static bool g_blackTried = false;
static bool g_blackAllowed = true;         // [XR] BlackSceneLayer=0 stands the placeholder down
// Static branded backplate behind startup/menu content only. Gameplay never submits it.
static const int32_t kBrandW = 1536, kBrandH = 1024;
static XrSwapchain g_brandSc = 0;
static std::vector<ID3D11Texture2D*> g_brandImgs;
static bool g_brandTried = false;
static bool g_brandReady = false;
static volatile long  g_brandOn = 1;
static volatile float g_cinemaDistanceM = 2.6f;
static volatile float g_cinemaWidthDeg = 74.0f;
static volatile long  g_cinemaHeadLocked = 0;
static long           g_thEnv = 1;              // [Cinema] Environment: the dark theater room
static long           g_thCurved = 1;           // [Cinema] Curved: cylinder screen when supported
static bool           g_cylinderExt = false;    // XR_KHR_composition_layer_cylinder enabled
static bool g_perEyePose = false;          // resolved from g_perEyeMode once the runtime is known
static int  g_perEyeMode = -1;             // [XR] PerEyePose: -1 auto, 0 off, 1 on
static bool g_declLocal = true;            // [LINKPOSE] (Beta 1.1) [XR] DeclareTruePose: projection poses in LOCAL
static bool g_fitFrustum = false;          // [FITFRUSTUM], resolved once the runtime is known
static int  g_fitMode = -1;                // [XR] FitRuntimeFrustum: -1 auto, 0 off, 1 on

// [SUBMITTHREAD] SteamVR parks its environment grid at ~1.0 alpha over any scene app it still
// considers "loading", and it clears that state only when the app delivers frames at the
// compositor's own cadence - measured live 2026-08-04 (grid pinned at 0.99 for whole sessions,
// actively re-asserted against a forced FadeGrid; Home clears it in ~3s). This game tops out
// near 45fps with culling force-disabled, so a submit loop driven by the game's Present can
// NEVER satisfy a 90Hz compositor and the grid never lifts - the "game is transparent over the
// SteamVR environment" report. Virtual Desktop's runtime has no such gate, which is why the
// same code always looked fine on VDXR.
// The fix is to decouple: the game thread only PUBLISHES frames (D3D9 readback -> CPU buffer,
// paired with the pose that frame was rendered with), and a worker thread owns every OpenXR
// call, heartbeating waitFrame/endFrame at the compositor's rate and re-submitting the held
// frame whenever the game has not produced a new one. [XR] SubmitThread=0 reverts to the old
// inline submit.
static bool            g_useWorker = true;
static HANDLE          g_workerThread = nullptr;
static volatile LONG   g_workerStop = 0;
static SRWLOCK         g_pubLock = SRWLOCK_INIT;
static volatile long   g_pubZcSlot = -1;   // [ZEROCOPY] slot of the published frame, -1 = pixels in g_pubPixels
static volatile long   g_pubExSlot = -1;   // [EXSHARE] the same for a D3D9Ex shared frame
static std::vector<uint8_t> g_pubPixels;   // tightly packed BGRA rows, g_pubW*4 pitch
static UINT            g_pubW = 0, g_pubH = 0, g_pubPitch = 0;
static uint64_t        g_pubSeq = 0;       // bumped once per published game frame
static XrPosef         g_pubPose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
static LONG            g_pubPoseValid = 0;
static uint64_t        g_conSeq = 0;       // last seq the worker uploaded into g_uploadTex
static XrPosef         g_conPose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
static LONG            g_conPoseValid = 0;
// [FOVLATCH] The rendered FOV must travel WITH the image, for the same reason the pose does.
// Outlast changes its camera FOV constantly (DefaultFOV 90, RunningFOV 100, camcorder zoom to
// 15), and the worker re-submits a HELD image on the compositor frames where the game has not
// produced a new one. Reading the live FOV on those frames re-declares the same pixels at a
// different angle, so the image visibly resizes between one compositor frame and the next -
// seen in the log as eyeFov flipping 88.0x94.7 <-> 87.8x94.5 mid-session, and felt as flicker
// while moving. Inline mode never had this: capture and FOV read happened in one Present.
static float           g_pubFovX = 0.0f, g_pubFovY = 0.0f;
static float           g_conFovX = 0.0f, g_conFovY = 0.0f;
// [POSEFREEZE] The worker locates a new head pose 90x/second, but the GAME reads that pose
// mid-frame, at whatever moment each camera call happens. Two failures come straight out of
// that, and both are "jitter only while the head is moving":
//   - Outlast builds its two eye views one after the other, so with a live pose stream the two
//     eyes can render from DIFFERENT head poses - the pair no longer fuses cleanly.
//   - The image's true pose is then no single value, so whatever the submit declares is wrong
//     by a variable amount, and the compositor's reprojection error changes every frame.
// Cure: the game only ever sees a pose FROZEN for the whole frame. The worker writes into the
// pending slot; the game thread promotes pending -> live exactly once per frame, at publish,
// after tagging the finished image with the live value it was actually rendered with. Both
// eyes identical by construction, and declared == rendered exactly.
static XrPosef         g_pendPose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
static float           g_pendYaw = 0.0f, g_pendPitch = 0.0f, g_pendRoll = 0.0f;
static LONG            g_pendValid = 0;
// [HANDSYNC] The same freeze for the hands: located at the head's display time, written beside
// the pending head pose, promoted with it. Arms are then solved against exactly the head pose
// the camera renders, instead of a pad sample one display period older that keeps moving
// mid-frame. Inline mode writes live directly, like the head.
static OLXrFrameHands  g_pendHands = {};
static LONG            g_pendHandsValid = 0;
static OLXrFrameHands  g_liveHands = {};
static LONG            g_liveHandsValid = 0;
// [FRAMETAG] (Test36) [POSEFREEZE] above assumed Present runs on the game thread. It does not:
// UE3 builds the views on the GAME thread and presents on its RENDER thread, which may be a whole
// frame behind. Promoting at Present therefore let the game thread build frame N+1 with frame N's
// pose (or switch pose between the two eyes), and tagged each image with the pose of the frame
// after it - declared != rendered by one frame of head motion, intermittently: jitter while the
// head moves. Now the GAME thread latches the pose once per frame (first consumer of the frame,
// OLXR_FrameLatch) and records {pose, FOV} when the frame's views are built (OLXR_FrameCommit);
// the render thread recognises the frame it is drawing by its view matrices
// (OLXR_RenderViewSeen) and Present tags the image with exactly that record.
// [XR] FramePoseLatch=0 restores the Test35 behaviour.
struct FrameTag { XrPosef pose; LONG poseValid; float fovX, fovY; long lensFit; float key[32]; };
static FrameTag        g_tagRing[8];
static long            g_tagNext = 0;                    // under g_pubLock; newest at g_tagNext-1
static volatile long   g_latchOn = 1;
static volatile long   g_latchArmed = 1;     // the next consumer on the game thread latches
static volatile long   g_latchLive = 0;      // Presents left before the old Present-time promote returns
static volatile long   g_tagStat[4] = { 0, 0, 0, 0 };   // Presents tagged, untagged; view lookups missed, hit
static volatile long   g_tagDiff = 0;        // tags the old rule would have got wrong
static volatile DWORD  g_tidGame = 0, g_tidPresent = 0;
// [LENSFIT] (Test36) the frame was rendered with each eye's own off-centre frustum (ol_ue3); the
// bridge then maps it 1:1 instead of cropping and stretching a symmetric picture.
static long            g_pubLensFit = 0, g_conLensFit = 0;
static volatile long   g_lensFitOn = 1;      // [Render] LensFit
static long            g_lensFitIni = 1;     // as the ini has it: 1 = on except Quest Link, 2 = on everywhere, 0 = off
static volatile long   g_fakeEyeFov = 0;     // [Debug] FakeEyeFov: Quest 3 frusta on any runtime
// [DEVBRIDGE] "hands 1": fixed controller poses in front of the body, for testing the arm solver
// on a headset with no controllers.
static volatile long   g_devFakeHands = 0;
static volatile long   g_devHeadWave = 0;   // [DEVBRIDGE] "headwave 1"
static volatile float  g_devLookYaw = 0.0f, g_devLookPitch = 0.0f;   // [DEVBRIDGE] "look <yaw> <pitch>" (degrees)
static XrVector3f      g_devHeadRel = { 0.0f, 0.0f, 0.0f };
static bool RunXrFrame(IDirect3DDevice9* dev, bool preCaptured);
static DWORD WINAPI WorkerMain(LPVOID);
// DEFAULT 0.4 (2026-07-23), matching ME1 (vr_config.h headLookSmoothing=0.4). At 90 fps and
// stable frame time raw tracking was fine, but exclusive-fullscreen 4K runs slower and with more
// frame-time jitter, and raw pose then reads as "not smooth". This is the smoothing STRENGTH fed
// to a speed-adaptive quaternion filter below (not the old per-axis Euler EMA): heavy at rest,
// disengaging to raw on fast turns, so it adds no lag when you actually look around.
static volatile float g_smoothing = 0.4f;
// BODY-BOUNDED 6DOF. Outlast's pawn does not follow room-scale headset translation, so an
// unrestricted OpenXR position lets the camera physically leave Blake's body during gameplay
// and authored scenes. Clamp the ONE shared position before it is consumed by either the UE3
// camera hook or the compositor. Keeping this upstream is essential: clamping only the game
// camera would make the submitted pose disagree with the rendered image and cause swimming.
// Coordinates are the recentered OpenXR frame: +X right, +Y up, -Z forward.
static volatile long  g_bodyAnchor = 1;
// [BODYFOLLOW] (Test39) The body comes with you: across the floor (not up/down - physical crouch
// reads height) the position baseline is dragged along once the head passes the body bounds, and
// drifts back under a head that has sat still in a new spot. Before, sitting back 15 cm pinned the
// view at the 8 cm bound for good and the body felt "off" until a recenter.
static volatile long  g_bodyFollow = 1;
static volatile float g_bodySideM = 0.12f;
static volatile float g_bodyForwardM = 0.10f;
static volatile float g_bodyBackwardM = 0.08f;
static volatile float g_bodyUpM = 0.18f;
static volatile float g_bodyDownM = 0.35f;
// [ROOMWALK] (Test44) Walk around your room and Miles walks with you. What [BODYFOLLOW] drags the
// position baseline by (the head past the body bounds, or the slow settle under a head that sat still)
// used to be thrown away: the view stopped at the bound and the body stayed put. Now, while the game
// thread says he may be walked (ol_interact, refreshed every frame), it is handed to Miles's feet: a
// running total rides with each pose it was taken from (pending -> latched with the pose at the frame
// latch), and the game thread moves him by the latched total at the start of the player tick - so his
// body and the view move in the same frame (no lag, no judder), through the engine's collision.
static volatile ULONGLONG g_rwAllowUntil = 0;
static double g_rwCum[2] = { 0.0, 0.0 };                                  // XR thread: handed so far (m, recenter x, z)
static double g_rwPend[2] = { 0.0, 0.0 }, g_rwLat[2] = { 0.0, 0.0 };      // with the pending / the latched pose (g_pubLock)
static double g_rwDone[2] = { 0.0, 0.0 };                                 // game thread: what it has taken
extern "C" void OLXR_RoomWalkAllow(int on) { g_rwAllowUntil = on ? GetTickCount64() + 200 : 0; }
// Frame-to-frame angle past which the filter fully disengages (ME1 kSnapSpeedDeg). A real head
// turn easily exceeds this, so turns are raw; a still head sits well under it, so jitter is
// filtered. Held per-eye-independent (one filter on the centre pose).
static const float kSnapDeg = 0.4f;
static XrQuaternionf g_smoothQuat = { 0,0,0,1 };
static bool g_smoothInit = false;
// [HEADPOS] Position is deliberately NOT smoothed. An earlier attempt (2026-08-17) ran it
// through this same filter to cure a reported clash with the rotation, and it read as worse;
// ME1's reference model filters the quaternion only and uses the raw view centre for lean.
extern "C" float OLXR_GetConvergence() { return g_convShift; }
extern "C" void  OLXR_SetConvergence(float v) {
    if (v < -3.0f) v = -3.0f; if (v > 3.0f) v = 3.0f;
    g_convShift = v;
}
extern "C" float OLXR_GetSmoothing() { return g_smoothing; }
extern "C" void  OLXR_SetSmoothing(float v) {
    if (v < 0.0f) v = 0.0f; if (v > 0.9f) v = 0.9f;
    g_smoothing = v;
}

// CINEMA SCREEN. Menus (~33 deg/eye) and scripted/cutscene cameras (~58 deg/eye) render a much
// narrower FOV than gameplay (~88 deg/eye after the OLGame.ini levers). The projection layer
// must declare exactly what was rendered, so those frames honestly occupy a small box in the
// middle of a ~94 deg headset - the "menu is tiny" complaint. A QUAD makes no claim about world
// geometry, so it is the one place magnification is legitimate: below this threshold the two eye
// images move onto a pair of per-eye quads (one image per eye - stereo survives, and no
// side-by-side double image, which is what sank the reverted whole-backbuffer attempt) sized to
// fill the headset. Camcorder-up stays projection (88 deg > threshold); only a deliberate deep
// camcorder zoom crosses it, and a zoom that fills the view is what a zoom is for.
// Threshold is in rendered per-eye HORIZONTAL degrees; 0 disables. Hysteresis avoids flapping
// while FOVApproach sweeps through the boundary.
// DEFAULT OFF (2026-07-23). A flat 2D screen overlay breaks presence - it uses the same large
// screen the menu uses, and reads as looking through a flat screen rather than being IN the
// world, which is the goal even when the game renders a narrow frame. Worse, with the
// screen enabled it silently ATE the failure of the [CINEFOV] widening: the fix did nothing and
// the screen made that look deliberate. Off means every 3D frame goes down the projection path
// with real stereo and real head tracking. The slider stays for anyone who wants it.
static float         g_screenBelowDeg = 0.0f;
static volatile long g_screenOn = 0;
// WORLD-LOCKED screen anchor (2026-07-23). An earlier attempt hung the screen in VIEW space -
// head-locked - while the camera injection was frozen, so NOTHING on screen responded to the
// head at all: read as head tracking not working in cutscenes. A cinema screen has to be a THEATER
// screen: fixed in the room (LOCAL space), planted where you were looking the moment it
// engaged, upright (yaw only). Turning your head then pans across and off the screen exactly
// like looking around a room - that is what reads as head tracking during cutscenes.
static XrPosef       g_screenPose = { {0,0,0,1}, {0,0,0} };
static long          g_screenPoseValid = 0;
// [THEATER] The world-locked menu/loading theater (see the [THEATER] block). Declared here so a
// recenter can ask for a fresh anchor; 0 = re-anchor in front of the head on the next frame.
static volatile long g_thValid = 0;
// [FADE] 0 = black .. 1 = full picture. Every change of presentation (loading panel, theater,
// stereo gameplay) cuts to black for one frame and fades the new picture in from here.
static float         g_presentFade = 1.0f;
static XrQuaternionf g_lastLocalQ = { 0,0,0,1 };   // raw LOCAL head pose from the last locate
static XrVector3f    g_lastLocalP = { 0,0,0 };
static long          g_lastLocalValid = 0;
// Read by the camera hook (ol_ue3.cpp): head rotation must NOT be injected while the screen is
// up - panning the picture inside a head-locked screen is the "warpy cutscene" look. ME1's cine
// answer is a tracking-off 3D picture; the injection weight ramps there, never steps.
extern "C" int   OLXR_ScreenActive() { return g_screenOn ? 1 : 0; }
extern "C" float OLXR_GetScreenBelow() { return g_screenBelowDeg; }
extern "C" void  OLXR_SetScreenBelow(float v) {
    if (v < 0.0f) v = 0.0f; if (v > 85.0f) v = 85.0f;
    g_screenBelowDeg = v;
}

// ---- [XRINPUT] motion controls: the OpenXR action set ------------------------------------------
// Step 1 proved the game reads a fabricated pad (empty slot 0, polled ~63/s, never gives up). This
// is the other half: read the VR controllers and fill that invented pad.
//
// THREADING, which is where this project keeps getting hurt: the submit worker owns EVERY OpenXR
// call, so xrSyncActions and the state reads happen THERE, right after xrWaitFrame (its predicted
// display time is what the pose locate wants). The game thread never touches OpenXR - it reads the
// published snapshot below under the same lock as the head pose. And the [POSEFREEZE] rule
// applies: the worker runs at 90 Hz against a ~60 fps game, so the game must see ONE coherent
// state per frame, not a value that changes mid-frame. g_padPub is written whole under the lock.
// OLXrPad / OLXR_BTN_* live in ol_xr.h - d3d9_proxy.cpp consumes them. A custom bitmask rather
// than XInput's, so the mapping to XInput lives in ONE table over there and a rebind never has
// to come back into this file.
static XrActionSet g_actionSet = 0;
static XrAction g_aMove = 0, g_aLook = 0, g_aTrig = 0, g_aGrip = 0, g_aPose = 0, g_aAim = 0;
static XrAction g_aClick = 0, g_aPrimary = 0, g_aSecondary = 0, g_aMenu = 0;
static XrAction g_aHaptic = 0;   // [HAPTICS] vibration output, both hands
static XrAction g_aStickTouch = 0;   // (Test48) [STICKCENTER] thumb on the stick (Touch / Index / Vive pad), optional
static XrPath   g_handPath[2] = { 0, 0 };
static XrSpace  g_handSpace[2] = { 0, 0 };
static XrSpace  g_aimSpace[2]  = { 0, 0 };

// ---- [HANDCAM] the camcorder is the camera, so aiming the camera IS aiming the camcorder -------
// Hold the controller up and you are looking
// through the camcorder you are holding. The mechanism is the one already proven for the head -
// ol_ue3 injects whatever is in g_olYawRad/Pitch/Roll and g_olHeadX/Y/Z, and ol_xr builds the
// DECLARED orientation from those same numbers. So handing the camcorder to the right hand needs
// no new camera code at all: while it is active, those globals carry the HAND's pose instead of
// the head's. Declared == rendered by construction, exactly as it is for head tracking.
// The AIM pose is used, not the grip pose: aim has a runtime-stable forward (-Z out of the
// controller, where you point it), which is what a camera lens follows. Grip is palm-relative and
// mirrored between hands.
// Hand tremor is real, so the hand orientation gets the same speed-adaptive filter the head uses,
// with its own state.
static volatile long  g_handCamOn = 0;        // [Input] HandCamcorder; off for the clean baseline
static volatile long  g_handCamActive = 0;    // set by the grip, from the XInput hook
static volatile float g_handCamStrength = 1.0f;
static XrQuaternionf  g_handSmoothQ = { 0, 0, 0, 1 };
static bool           g_handSmoothInit = false;
static XrQuaternionf  g_handRelQ = { 0, 0, 0, 1 };   // aim orientation in the recenter frame
static XrVector3f     g_handRelP = { 0, 0, 0 };      // aim position in the recenter frame
static long           g_handRelValid = 0;
extern "C" void OLXR_SetHandCamActive(int on) { InterlockedExchange(&g_handCamActive, on ? 1 : 0); }
extern "C" int  OLXR_GetHandCam() { return g_handCamOn ? 1 : 0; }
extern "C" void OLXR_SetHandCam(int on) { g_handCamOn = on ? 1 : 0; }
extern "C" float OLXR_GetHandCamStrength() { return g_handCamStrength; }
extern "C" void  OLXR_SetHandCamStrength(float v) {
    if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
    g_handCamStrength = v;
}
static bool     g_inputReady = false;       // action set built AND attached
static long     g_inputEnabled = 1;         // [Input] MotionControls, ini
static OLXrPad  g_padPub = {};              // published snapshot, guarded by g_pubLock
static long     g_padPubValid = 0;
static long     g_inputLoggedBind = 0;

// ---- [HAPTICS] the game's own rumble, felt in the hands ----------------------------------------
// Outlast imports XInputSetState (xinput1_3 ordinal 3) and drives it from UE3's force-feedback
// waveforms. d3d9_proxy hooks it and hands the two motor levels here; the WORKER, the only thread
// that calls OpenXR, forwards them. A gamepad's two motors are a heavy and a light weight, not a
// left and a right hand, so both hands get the stronger of the two. Every pulse is short and is
// refreshed while the game keeps the motors running, so if the game stops talking (pause,
// alt-tab, a crash) the controllers go quiet by themselves instead of buzzing on.
static volatile long  g_hapticsOn = 1;           // [Input] Haptics
static volatile float g_hapticStrength = 1.0f;   // [Input] HapticStrength, 0..1.5
static volatile float g_rumbleLow = 0.0f, g_rumbleHigh = 0.0f;
static volatile ULONGLONG g_rumbleMs = 0;        // GetTickCount64 of the game's last SetState
static volatile long  g_rumbleSeen = 0;
extern "C" void OLXR_SetRumble(float lowFreq, float highFreq) {
    g_rumbleLow = lowFreq; g_rumbleHigh = highFreq; g_rumbleMs = GetTickCount64();
    if ((lowFreq > 0.0f || highFreq > 0.0f) && InterlockedCompareExchange(&g_rumbleSeen, 1, 0) == 0)
        XLOG("[OLVR][HAPTICS] first game rumble seen (low=%.2f high=%.2f) - forwarding to the controllers",
             lowFreq, highFreq);
}
extern "C" int   OLXR_GetHaptics() { return g_hapticsOn ? 1 : 0; }
extern "C" void  OLXR_SetHaptics(int on) { g_hapticsOn = on ? 1 : 0; }
extern "C" float OLXR_GetHapticStrength() { return g_hapticStrength; }
extern "C" void  OLXR_SetHapticStrength(float v) {
    if (!(v >= 0.0f)) v = 0.0f; if (v > 1.5f) v = 1.5f;
    g_hapticStrength = v;
}

// The quaternion helpers live further down (they belong with the pose code); the hand-pose
// transform here needs them, so declare rather than reorder a working file.
static XrQuaternionf QNormalize(XrQuaternionf q);
static XrQuaternionf QConj(XrQuaternionf q);
static XrQuaternionf QMul(XrQuaternionf a, XrQuaternionf b);
static XrVector3f QRotate(XrQuaternionf q, XrVector3f v);

static XrPath XrPathOf(const char* s) {
    XrPath p = XR_NULL_PATH_VALUE;
    if (g_fn.stringToPath) g_fn.stringToPath(g_instance, s, &p);
    return p;
}

// (Test48) Which controller layout the runtime picked for each hand. SteamVR maps a controller it has no
// layout of its own for onto one of the five this mod offers; a tester's Pimax controllers turned and walked
// Miles by themselves, so the log has to say which layout they ended up on.
static void LogInteractionProfiles(const char* why) {
    if (!g_fn.getCurrentInteractionProfile || !g_session || !g_handPath[0]) return;
    static char s_last[2][256] = {};   // SteamVR sends the change event several times in a row: log real changes
    for (int i = 0; i < 2; ++i) {
        XrInteractionProfileState st = {}; st.type = XR_TYPE_INTERACTION_PROFILE_STATE_VALUE;
        char name[256] = "none";
        const XrResult r = g_fn.getCurrentInteractionProfile(g_session, g_handPath[i], &st);
        if (XrSucceeded(r) && st.interactionProfile && g_fn.pathToString) {
            uint32_t n = 0;
            if (!XrSucceeded(g_fn.pathToString(g_instance, st.interactionProfile, (uint32_t)sizeof(name), &n, name))) strcpy_s(name, "?");
        } else if (!XrSucceeded(r)) sprintf_s(name, "unknown (r=%d)", r);
        if (!strcmp(name, s_last[i]) && strcmp(why, "first input") != 0) continue;
        strcpy_s(s_last[i], name);
        XLOG("[OLVR][XRINPUT] %s: %s hand uses %s", why, i ? "right" : "left", name);
    }
}

// Build the action set. Called once, after the session exists and BEFORE the first xrSyncActions:
// attach is a one-shot per session and every action must exist before it. A failure anywhere here
// is non-fatal by design - g_inputReady stays false, the pad publisher never runs, and the mod is
// exactly the build that works today.
static void BuildActionSet() {
    if (!g_inputEnabled || g_inputReady) return;
    if (!g_fn.createActionSet || !g_fn.createAction || !g_fn.attachSessionActionSets ||
        !g_fn.suggestInteractionProfileBindings || !g_fn.stringToPath) {
        XLOG("[OLVR][XRINPUT] this runtime did not provide the input entry points - motion controls off");
        return;
    }
    XrActionSetCreateInfo asci = {}; asci.type = XR_TYPE_ACTION_SET_CREATE_INFO_VALUE;
    strcpy_s(asci.actionSetName, "gameplay");
    strcpy_s(asci.localizedActionSetName, "Gameplay");
    asci.priority = 0;
    XrResult r = g_fn.createActionSet(g_instance, &asci, &g_actionSet);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XRINPUT] xrCreateActionSet r=%d", r); return; }

    g_handPath[0] = XrPathOf("/user/hand/left");
    g_handPath[1] = XrPathOf("/user/hand/right");

    // Every action is subaction-pathed to both hands, so one action serves left and right and the
    // state reads pick a side. Fewer actions, and the bindings stay symmetric.
    auto mk = [&](XrAction& out, XrActionType t, const char* name, const char* loc) -> bool {
        XrActionCreateInfo aci = {}; aci.type = XR_TYPE_ACTION_CREATE_INFO_VALUE;
        aci.actionType = t;
        strcpy_s(aci.actionName, name);
        strcpy_s(aci.localizedActionName, loc);
        aci.countSubactionPaths = 2; aci.subactionPaths = g_handPath;
        XrResult rr = g_fn.createAction(g_actionSet, &aci, &out);
        if (!XrSucceeded(rr)) { XLOG("[OLVR][XRINPUT] xrCreateAction('%s') r=%d", name, rr); return false; }
        return true;
    };
    bool ok = mk(g_aMove,      XR_ACTION_TYPE_VECTOR2F_INPUT_VALUE, "stick",      "Thumbstick")
           && mk(g_aTrig,      XR_ACTION_TYPE_FLOAT_INPUT_VALUE,    "trigger",    "Trigger")
           && mk(g_aGrip,      XR_ACTION_TYPE_FLOAT_INPUT_VALUE,    "grip",       "Grip")
           && mk(g_aClick,     XR_ACTION_TYPE_BOOLEAN_INPUT_VALUE,  "stickclick", "Thumbstick Click")
           && mk(g_aPrimary,   XR_ACTION_TYPE_BOOLEAN_INPUT_VALUE,  "primary",    "Primary Button")
           && mk(g_aSecondary, XR_ACTION_TYPE_BOOLEAN_INPUT_VALUE,  "secondary",  "Secondary Button")
           && mk(g_aMenu,      XR_ACTION_TYPE_BOOLEAN_INPUT_VALUE,  "menu",       "Menu Button")
           && mk(g_aPose,      XR_ACTION_TYPE_POSE_INPUT_VALUE,     "handpose",   "Hand Pose")
           && mk(g_aAim,       XR_ACTION_TYPE_POSE_INPUT_VALUE,     "aimpose",    "Aim Pose");
    if (!ok) return;
    g_aLook = g_aMove;   // one stick action; the right hand's copy IS the look stick
    // [HAPTICS] Optional output action. A runtime that refuses it leaves g_aHaptic at 0, the
    // binding tables skip it, and every input above is unaffected.
    if (!mk(g_aHaptic, XR_ACTION_TYPE_VIBRATION_OUTPUT_VALUE, "rumble", "Rumble")) g_aHaptic = 0;
    // (Test48) [STICKCENTER] optional the same way: refused, only the stick-drift fix is lost.
    if (!mk(g_aStickTouch, XR_ACTION_TYPE_BOOLEAN_INPUT_VALUE, "sticktouch", "Thumbstick Touch")) g_aStickTouch = 0;

    struct Bind { XrAction a; const char* path; };
    auto suggest = [&](const char* profile, const Bind* b, int n) {
        XrPath prof = XrPathOf(profile);
        if (!prof) return;
        XrActionSuggestedBinding sb[24];
        int m = 0;
        for (int i = 0; i < n && m < 24; ++i) {
            if (!b[i].a) continue;   // an optional action (haptics) the runtime refused to create
            XrPath p = XrPathOf(b[i].path);
            if (p) { sb[m].action = b[i].a; sb[m].binding = p; ++m; }
        }
        XrInteractionProfileSuggestedBinding ipb = {};
        ipb.type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING_VALUE;
        ipb.interactionProfile = prof; ipb.countSuggestedBindings = (uint32_t)m; ipb.suggestedBindings = sb;
        XrResult rr = g_fn.suggestInteractionProfileBindings(g_instance, &ipb);
        XLOG("[OLVR][XRINPUT] bindings %s -> r=%d (%d paths)", profile, rr, m);
    };

    // Oculus Touch: X/Y on the left controller, A/B on the right, menu on the left only.
    const Bind touch[] = {
        { g_aPose, "/user/hand/left/input/grip/pose" },   { g_aPose, "/user/hand/right/input/grip/pose" },
        { g_aAim,  "/user/hand/left/input/aim/pose" },    { g_aAim,  "/user/hand/right/input/aim/pose" },
        { g_aMove, "/user/hand/left/input/thumbstick" },  { g_aMove, "/user/hand/right/input/thumbstick" },
        { g_aClick,"/user/hand/left/input/thumbstick/click" }, { g_aClick,"/user/hand/right/input/thumbstick/click" },
        { g_aTrig, "/user/hand/left/input/trigger/value" }, { g_aTrig, "/user/hand/right/input/trigger/value" },
        { g_aGrip, "/user/hand/left/input/squeeze/value" }, { g_aGrip, "/user/hand/right/input/squeeze/value" },
        { g_aPrimary,   "/user/hand/left/input/x/click" }, { g_aPrimary,   "/user/hand/right/input/a/click" },
        { g_aSecondary, "/user/hand/left/input/y/click" }, { g_aSecondary, "/user/hand/right/input/b/click" },
        { g_aMenu, "/user/hand/left/input/menu/click" },
        { g_aHaptic, "/user/hand/left/output/haptic" }, { g_aHaptic, "/user/hand/right/output/haptic" },
        { g_aStickTouch, "/user/hand/left/input/thumbstick/touch" }, { g_aStickTouch, "/user/hand/right/input/thumbstick/touch" },
    };
    suggest("/interaction_profiles/oculus/touch_controller", touch, (int)(sizeof(touch)/sizeof(touch[0])));

    // Index: A/B on both controllers, squeeze force is a separate path from the analog grip.
    const Bind index[] = {
        { g_aPose, "/user/hand/left/input/grip/pose" },   { g_aPose, "/user/hand/right/input/grip/pose" },
        { g_aAim,  "/user/hand/left/input/aim/pose" },    { g_aAim,  "/user/hand/right/input/aim/pose" },
        { g_aMove, "/user/hand/left/input/thumbstick" },  { g_aMove, "/user/hand/right/input/thumbstick" },
        { g_aClick,"/user/hand/left/input/thumbstick/click" }, { g_aClick,"/user/hand/right/input/thumbstick/click" },
        { g_aTrig, "/user/hand/left/input/trigger/value" }, { g_aTrig, "/user/hand/right/input/trigger/value" },
        { g_aGrip, "/user/hand/left/input/squeeze/value" }, { g_aGrip, "/user/hand/right/input/squeeze/value" },
        { g_aPrimary,   "/user/hand/left/input/a/click" }, { g_aPrimary,   "/user/hand/right/input/a/click" },
        { g_aSecondary, "/user/hand/left/input/b/click" }, { g_aSecondary, "/user/hand/right/input/b/click" },
        { g_aMenu, "/user/hand/left/input/system/click" },
        { g_aHaptic, "/user/hand/left/output/haptic" }, { g_aHaptic, "/user/hand/right/output/haptic" },
        { g_aStickTouch, "/user/hand/left/input/thumbstick/touch" }, { g_aStickTouch, "/user/hand/right/input/thumbstick/touch" },
    };
    suggest("/interaction_profiles/valve/index_controller", index, (int)(sizeof(index)/sizeof(index[0])));

    // WMR: trackpad AND stick both exist; bind the stick, it is what people use.
    const Bind wmr[] = {
        { g_aPose, "/user/hand/left/input/grip/pose" },   { g_aPose, "/user/hand/right/input/grip/pose" },
        { g_aAim,  "/user/hand/left/input/aim/pose" },    { g_aAim,  "/user/hand/right/input/aim/pose" },
        { g_aMove, "/user/hand/left/input/thumbstick" },  { g_aMove, "/user/hand/right/input/thumbstick" },
        { g_aClick,"/user/hand/left/input/thumbstick/click" }, { g_aClick,"/user/hand/right/input/thumbstick/click" },
        { g_aTrig, "/user/hand/left/input/trigger/value" }, { g_aTrig, "/user/hand/right/input/trigger/value" },
        { g_aGrip, "/user/hand/left/input/squeeze/click" }, { g_aGrip, "/user/hand/right/input/squeeze/click" },
        { g_aMenu, "/user/hand/left/input/menu/click" },
        { g_aHaptic, "/user/hand/left/output/haptic" }, { g_aHaptic, "/user/hand/right/output/haptic" },
    };
    suggest("/interaction_profiles/microsoft/motion_controller", wmr, (int)(sizeof(wmr)/sizeof(wmr[0])));

    // Vive wands: no thumbstick at all, the trackpad stands in for it.
    const Bind vive[] = {
        { g_aPose, "/user/hand/left/input/grip/pose" },   { g_aPose, "/user/hand/right/input/grip/pose" },
        { g_aAim,  "/user/hand/left/input/aim/pose" },    { g_aAim,  "/user/hand/right/input/aim/pose" },
        { g_aMove, "/user/hand/left/input/trackpad" },    { g_aMove, "/user/hand/right/input/trackpad" },
        { g_aClick,"/user/hand/left/input/trackpad/click" }, { g_aClick,"/user/hand/right/input/trackpad/click" },
        { g_aStickTouch, "/user/hand/left/input/trackpad/touch" }, { g_aStickTouch, "/user/hand/right/input/trackpad/touch" },
        { g_aTrig, "/user/hand/left/input/trigger/value" }, { g_aTrig, "/user/hand/right/input/trigger/value" },
        { g_aGrip, "/user/hand/left/input/squeeze/click" }, { g_aGrip, "/user/hand/right/input/squeeze/click" },
        { g_aMenu, "/user/hand/left/input/menu/click" },
        { g_aHaptic, "/user/hand/left/output/haptic" }, { g_aHaptic, "/user/hand/right/output/haptic" },
    };
    suggest("/interaction_profiles/htc/vive_controller", vive, (int)(sizeof(vive)/sizeof(vive[0])));

    // Simple controller: the fallback every runtime must support. Pose + select + menu only,
    // so an unknown controller still gets SOMETHING rather than a dead action set.
    const Bind simple[] = {
        { g_aPose, "/user/hand/left/input/grip/pose" },   { g_aPose, "/user/hand/right/input/grip/pose" },
        { g_aAim,  "/user/hand/left/input/aim/pose" },    { g_aAim,  "/user/hand/right/input/aim/pose" },
        { g_aTrig, "/user/hand/left/input/select/click" },{ g_aTrig, "/user/hand/right/input/select/click" },
        { g_aMenu, "/user/hand/left/input/menu/click" },
        { g_aHaptic, "/user/hand/left/output/haptic" }, { g_aHaptic, "/user/hand/right/output/haptic" },
    };
    suggest("/interaction_profiles/khr/simple_controller", simple, (int)(sizeof(simple)/sizeof(simple[0])));

    // Pose spaces, two per hand: GRIP (where the hand is) and AIM (where it points). Non-fatal -
    // without them the sticks and buttons still work, only the hand-held camcorder is lost.
    if (g_fn.createActionSpace) {
        for (int i = 0; i < 2; ++i) {
            XrActionSpaceCreateInfo sp = {}; sp.type = XR_TYPE_ACTION_SPACE_CREATE_INFO_VALUE;
            sp.action = g_aPose; sp.subactionPath = g_handPath[i]; sp.poseInActionSpace = IdentityPose();
            if (!XrSucceeded(g_fn.createActionSpace(g_session, &sp, &g_handSpace[i]))) g_handSpace[i] = 0;
            XrActionSpaceCreateInfo ap = {}; ap.type = XR_TYPE_ACTION_SPACE_CREATE_INFO_VALUE;
            ap.action = g_aAim; ap.subactionPath = g_handPath[i]; ap.poseInActionSpace = IdentityPose();
            if (!XrSucceeded(g_fn.createActionSpace(g_session, &ap, &g_aimSpace[i]))) g_aimSpace[i] = 0;
        }
    }

    XrSessionActionSetsAttachInfo at = {}; at.type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO_VALUE;
    at.countActionSets = 1; at.actionSets = &g_actionSet;
    r = g_fn.attachSessionActionSets(g_session, &at);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XRINPUT] xrAttachSessionActionSets r=%d - motion controls off", r); return; }
    g_inputReady = true;
    XLOG("[OLVR][XRINPUT] action set attached - VR controllers are live");
}

// Read one action's state. Every getter is null-checked; a runtime that only half-implements
// input degrades field by field instead of taking the session down.
static float ReadFloat(XrAction a, int hand) {
    if (!g_fn.getActionStateFloat) return 0.0f;
    XrActionStateGetInfo gi = {}; gi.type = XR_TYPE_ACTION_STATE_GET_INFO_VALUE;
    gi.action = a; gi.subactionPath = g_handPath[hand];
    XrActionStateFloat st = {}; st.type = XR_TYPE_ACTION_STATE_FLOAT_VALUE;
    if (!XrSucceeded(g_fn.getActionStateFloat(g_session, &gi, &st)) || !st.isActive) return 0.0f;
    return st.currentState;
}
static bool ReadBool(XrAction a, int hand) {
    if (!g_fn.getActionStateBoolean) return false;
    XrActionStateGetInfo gi = {}; gi.type = XR_TYPE_ACTION_STATE_GET_INFO_VALUE;
    gi.action = a; gi.subactionPath = g_handPath[hand];
    XrActionStateBoolean st = {}; st.type = XR_TYPE_ACTION_STATE_BOOLEAN_VALUE;
    if (!XrSucceeded(g_fn.getActionStateBoolean(g_session, &gi, &st)) || !st.isActive) return false;
    return st.currentState != 0;
}
static void ReadStick(XrAction a, int hand, float* x, float* y) {
    *x = 0.0f; *y = 0.0f;
    if (!g_fn.getActionStateVector2f) return;
    XrActionStateGetInfo gi = {}; gi.type = XR_TYPE_ACTION_STATE_GET_INFO_VALUE;
    gi.action = a; gi.subactionPath = g_handPath[hand];
    XrActionStateVector2f st = {}; st.type = XR_TYPE_ACTION_STATE_VECTOR2F_VALUE;
    if (!XrSucceeded(g_fn.getActionStateVector2f(g_session, &gi, &st)) || !st.isActive) return;
    *x = st.currentState.x; *y = st.currentState.y;
}

// (Test48) [STICKCENTER] A tester on a Pimax headset (SteamVR): "when out of the car the image spins around like
// crazy", and with the stick unbound in SteamVR he could not walk. A stick that does not read 0 at rest turns or
// walks Miles by itself - the game takes the value as it is. Where the controller reports a thumb on the stick
// (Touch, Index, the Vive pad), what the stick reads with NO thumb on it is drift: once that is seen (over 0.10
// for half a second), the stick gives nothing while no thumb is on it, and its resting point is taken off while
// one is. A healthy stick is never changed. A touch sensor that reports no thumb while the stick swings is
// ignored from then on. [Input] StickAutoCenter=0 turns this off; [Input] StickDeadzone (radial, 0.00 - 0.60,
// default 0 = the game's own) catches drift on controllers with no touch sensor.
static volatile long  g_stickAutoCenter = 1;
static volatile float g_stickDeadzone = 0.0f;
static float s_stickRest[2][2] = {};
static bool  s_stickDrift[2] = { false, false }, s_touchBroken[2] = { false, false };
static int   s_stickTouch[2] = { -1, -1 };   // last sample: -1 no sensor, 0 no thumb, 1 thumb on it
static void StickCenter(int hand, float* x, float* y) {
    const float rx = *x, ry = *y, mag = sqrtf(rx * rx + ry * ry);
    const ULONGLONG now = GetTickCount64();
    int touch = -1;
    if (g_stickAutoCenter && g_aStickTouch && !s_touchBroken[hand] && g_fn.getActionStateBoolean) {
        XrActionStateGetInfo gi = {}; gi.type = XR_TYPE_ACTION_STATE_GET_INFO_VALUE;
        gi.action = g_aStickTouch; gi.subactionPath = g_handPath[hand];
        XrActionStateBoolean st = {}; st.type = XR_TYPE_ACTION_STATE_BOOLEAN_VALUE;
        if (XrSucceeded(g_fn.getActionStateBoolean(g_session, &gi, &st)) && st.isActive) touch = st.currentState ? 1 : 0;
    }
    static float s_lo[2][2], s_hi[2][2];
    static ULONGLONG s_win[2] = { 0, 0 }, s_driftSince[2] = { 0, 0 };
    if (touch == 0) {
        if (!s_win[hand] || now - s_win[hand] > 1000) { s_win[hand] = now; s_lo[hand][0] = s_hi[hand][0] = rx; s_lo[hand][1] = s_hi[hand][1] = ry; }
        if (rx < s_lo[hand][0]) s_lo[hand][0] = rx; if (rx > s_hi[hand][0]) s_hi[hand][0] = rx;
        if (ry < s_lo[hand][1]) s_lo[hand][1] = ry; if (ry > s_hi[hand][1]) s_hi[hand][1] = ry;
        if (s_hi[hand][0] - s_lo[hand][0] > 0.5f || s_hi[hand][1] - s_lo[hand][1] > 0.5f) {
            s_touchBroken[hand] = true; s_stickDrift[hand] = false; s_stickRest[hand][0] = s_stickRest[hand][1] = 0.0f; touch = -1;
            XLOG("[OLVR][STICKCENTER] the %s stick swung with no thumb reported on it - its touch sensor is not used", hand ? "right" : "left");
        } else {
            s_stickRest[hand][0] += (rx - s_stickRest[hand][0]) * 0.08f;
            s_stickRest[hand][1] += (ry - s_stickRest[hand][1]) * 0.08f;
            if (mag > 0.10f) { if (!s_driftSince[hand]) s_driftSince[hand] = now; }
            else s_driftSince[hand] = 0;
            if (!s_stickDrift[hand] && s_driftSince[hand] && now - s_driftSince[hand] > 500) {
                s_stickDrift[hand] = true;
                XLOG("[OLVR][STICKCENTER] the %s stick reads (%.2f %.2f) with no thumb on it - drift: it gives nothing until touched, and its resting point is taken off",
                     hand ? "right" : "left", rx, ry);
            }
        }
    } else s_win[hand] = 0;
    s_stickTouch[hand] = touch;
    float ox = rx, oy = ry;
    if (s_stickDrift[hand] && touch >= 0) {
        if (touch == 0) ox = oy = 0.0f;
        else { ox -= s_stickRest[hand][0]; oy -= s_stickRest[hand][1]; }
    }
    const float dz = g_stickDeadzone;
    if (dz > 0.0f) {
        const float m = sqrtf(ox * ox + oy * oy);
        if (m <= dz) ox = oy = 0.0f;
        else { const float k = (m - dz) / (1.0f - dz) / m; ox *= k; oy *= k; }
    }
    *x = ox; *y = oy;
}

// [HAPTICS] WORKER THREAD ONLY, once per frame after xrSyncActions. XInput motor levels persist
// until the game changes them, so the last level stays on whether the game re-sends it every
// tick or only on change - with a 1 s cap, so a level nobody refreshes cannot buzz forever.
// Re-sends on a real change or every 60 ms; each pulse lasts 100 ms, so a held rumble never gaps,
// and when the session loses focus (this function stops running) the controllers fall silent.
// [INTERACT] (Test37) Short per-hand pulses (a grab, a battery click), on top of the game's rumble.
static volatile float     g_pulseAmp[2] = { 0.0f, 0.0f };
static volatile ULONGLONG g_pulseUntil[2] = { 0, 0 };
extern "C" void OLXR_HapticPulse(int hand, float amp, int ms) {
    if (hand < 0 || hand > 1 || !g_hapticsOn) return;
    g_pulseAmp[hand] = amp > 1.0f ? 1.0f : amp;
    g_pulseUntil[hand] = GetTickCount64() + (ULONGLONG)(ms > 0 ? ms : 20);
}
static void ApplyHaptics() {
    if (!g_aHaptic || !g_fn.applyHapticFeedback) return;
    static float s_sent[2] = { 0.0f, 0.0f };
    static DWORD s_sentMs[2] = { 0, 0 };
    const DWORD now = GetTickCount();
    const ULONGLONG now64 = GetTickCount64();
    float rumble = 0.0f;
    const ULONGLONG rumbleMs = g_rumbleMs;
    if (g_hapticsOn && rumbleMs && now64 - rumbleMs < 1000) {
        const float lo = g_rumbleLow, hi = g_rumbleHigh;
        rumble = (lo > hi ? lo : hi) * g_hapticStrength;
        if (rumble > 1.0f) rumble = 1.0f;
        if (rumble < 0.02f) rumble = 0.0f;
    }
    XrHapticActionInfo info = {}; info.type = XR_TYPE_HAPTIC_ACTION_INFO_VALUE; info.action = g_aHaptic;
    for (int h = 0; h < 2; ++h) {
        float amp = rumble;
        if (now64 < g_pulseUntil[h] && g_pulseAmp[h] > amp) amp = g_pulseAmp[h];
        info.subactionPath = g_handPath[h];
        if (amp <= 0.0f) {
            if (s_sent[h] > 0.0f && g_fn.stopHapticFeedback) g_fn.stopHapticFeedback(g_session, &info);
            s_sent[h] = 0.0f;
            continue;
        }
        if (fabsf(amp - s_sent[h]) < 0.03f && (now - s_sentMs[h]) < 60) continue;
        XrHapticVibration v = {}; v.type = XR_TYPE_HAPTIC_VIBRATION_VALUE;
        v.duration = 100000000;   // 100 ms, nanoseconds
        v.frequency = XR_FREQUENCY_UNSPECIFIED_VALUE;
        v.amplitude = amp;
        g_fn.applyHapticFeedback(g_session, &info, (const XrHapticBaseHeader*)&v);
        s_sent[h] = amp; s_sentMs[h] = now;
    }
}

// [DEVBRIDGE] Scripted controllers for testing on a headset that has none. Each hand sits at an
// offset from the head (recenter frame, metres: +x right, +y up, -z forward) and glides to a new
// one over a set time ("hand L -0.25 -0.55 0.05 400"); "grip L 1" closes that hand. With no script
// the hands rest in front of the chest and the right one slowly rises and falls ("hands 1").
static SRWLOCK g_devHandLock = SRWLOCK_INIT;
struct DevHand { float from[3], to[3]; ULONGLONG t0, t1; bool scripted; float grip; float q[4]; };
static DevHand g_devHand[2] = {
    { { -0.20f, -0.38f, -0.32f }, { -0.20f, -0.38f, -0.32f }, 0, 0, false, 0.0f, { 0, 0, 0, 1 } },
    { {  0.20f, -0.38f, -0.32f }, {  0.20f, -0.38f, -0.32f }, 0, 0, false, 0.0f, { 0, 0, 0, 1 } } };
// (Test40) "hrot R pitch yaw roll": the scripted controller's angle (degrees, recenter frame: pitch up,
// yaw left, roll clockwise as you look along it), to stand in for how a real hand holds it.
extern "C" void OLXR_DevHandRot(int hand, float pitch, float yaw, float roll) {
    if (hand < 0 || hand > 1) return;
    const float hp = pitch * 0.00872664626f, hy = yaw * 0.00872664626f, hr = -roll * 0.00872664626f;
    const float qy[4] = { 0, sinf(hy), 0, cosf(hy) }, qp[4] = { sinf(hp), 0, 0, cosf(hp) }, qr[4] = { 0, 0, sinf(hr), cosf(hr) };
    auto mul = [](const float a[4], const float b[4], float o[4]) {
        o[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
        o[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
        o[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
        o[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
    };
    float t[4], q[4]; mul(qy, qp, t); mul(t, qr, q);
    AcquireSRWLockExclusive(&g_devHandLock);
    for (int i = 0; i < 4; ++i) g_devHand[hand].q[i] = q[i];
    ReleaseSRWLockExclusive(&g_devHandLock);
}
extern "C" void OLXR_DevHandTo(int hand, float x, float y, float z, int ms) {
    if (hand < 0 || hand > 1) return;
    AcquireSRWLockExclusive(&g_devHandLock);
    DevHand& h = g_devHand[hand];
    const ULONGLONG now = GetTickCount64();
    // Start from wherever the hand is right now, so a new move never jumps.
    float cur[3];
    const float a = (h.t1 > h.t0 && now < h.t1) ? (float)(now - h.t0) / (float)(h.t1 - h.t0) : 1.0f;
    for (int i = 0; i < 3; ++i) cur[i] = h.from[i] + (h.to[i] - h.from[i]) * a;
    for (int i = 0; i < 3; ++i) h.from[i] = cur[i];
    h.to[0] = x; h.to[1] = y; h.to[2] = z;
    h.t0 = now; h.t1 = now + (ULONGLONG)(ms > 0 ? ms : 1);
    h.scripted = true;
    ReleaseSRWLockExclusive(&g_devHandLock);
}
extern "C" void OLXR_DevGrip(int hand, float v) {
    if (hand < 0 || hand > 1) return;
    g_devHand[hand].grip = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
// [DEVBRIDGE] (Test39) scripted controller buttons / triggers, merged like the scripted grips, so a
// test press goes through the mod's own Quest -> pad mapping (chords, held-press extension).
static volatile unsigned g_devBtnMask = 0;
static volatile ULONGLONG g_devBtnUntil = 0;
static volatile float g_devTrig[2] = { 0.0f, 0.0f };
extern "C" void OLXR_DevButtons(unsigned mask, int ms) { g_devBtnMask = mask; g_devBtnUntil = GetTickCount64() + (ULONGLONG)(ms > 0 ? ms : 120); }
extern "C" void OLXR_DevTrigger(int hand, float v) { if (hand >= 0 && hand < 2) g_devTrig[hand] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
// [DEVBRIDGE] (Test46) "vrstick lx ly rx ry ms": the controllers' own sticks, through the mod's turning / zoom
static volatile float g_devStick[4] = { 0, 0, 0, 0 };
static volatile ULONGLONG g_devStickUntil = 0;
extern "C" void OLXR_DevStick(float lx, float ly, float rx, float ry, int ms) { g_devStick[0] = lx; g_devStick[1] = ly; g_devStick[2] = rx; g_devStick[3] = ry; g_devStickUntil = GetTickCount64() + (ULONGLONG)(ms > 0 ? ms : 200); }
// [DEVBRIDGE] "headoff": a head that moved (recenter frame, metres) on a headset that never does
static volatile float g_devHeadOff[3] = { 0.0f, 0.0f, 0.0f };
extern "C" void OLXR_DevHeadOffset(float x, float y, float z) { g_devHeadOff[0] = x; g_devHeadOff[1] = y; g_devHeadOff[2] = z; }
static void DevFakeHands(OLXrPose hand[2], unsigned* valid, XrTime t) {
    const float s = sinf((float)((double)(t % 4000000000LL) * 1e-9 * 3.14159265 * 0.5));
    const ULONGLONG now = GetTickCount64();
    AcquireSRWLockShared(&g_devHandLock);
    for (int i = 0; i < 2; ++i) {
        const DevHand& h = g_devHand[i];
        float o[3];
        const float a = (h.t1 > h.t0 && now < h.t1) ? (float)(now - h.t0) / (float)(h.t1 - h.t0) : 1.0f;
        const float e = a * a * (3.0f - 2.0f * a);   // ease in/out: a hand accelerates and settles
        for (int k = 0; k < 3; ++k) o[k] = h.from[k] + (h.to[k] - h.from[k]) * e;
        if (!h.scripted && i == 1) o[1] += 0.18f * s;
        hand[i].qx = h.q[0]; hand[i].qy = h.q[1]; hand[i].qz = h.q[2]; hand[i].qw = h.q[3];
        hand[i].px = g_devHeadRel.x + o[0];
        hand[i].py = g_devHeadRel.y + o[1];
        hand[i].pz = g_devHeadRel.z + o[2];
    }
    ReleaseSRWLockShared(&g_devHandLock);
    *valid = 3u;
}

// Sync + publish. WORKER THREAD ONLY, once per xrWaitFrame. The whole struct is built locally
// and copied under the lock in one go, so the game thread can never see a half-updated pad.
static void SyncInput(XrTime displayTime) {
    if (!g_inputReady || !g_fn.syncActions) return;
    XrActiveActionSet aas = {}; aas.actionSet = g_actionSet; aas.subactionPath = XR_NULL_PATH_VALUE;
    XrActionsSyncInfo si = {}; si.type = XR_TYPE_ACTIONS_SYNC_INFO_VALUE;
    si.countActiveActionSets = 1; si.activeActionSets = &aas;
    const XrResult r = g_fn.syncActions(g_session, &si);
    // XR_SESSION_NOT_FOCUSED is normal and frequent (dashboard up, app in the background) - it
    // is not an error, it just means every action reads inactive this frame.
    if (!XrSucceeded(r)) return;

    OLXrPad p = {};
    ReadStick(g_aMove, 0, &p.moveX, &p.moveY);
    ReadStick(g_aLook, 1, &p.lookX, &p.lookY);
    // (Test48) [STICKCENTER] raw values first, for the log below
    const float rawStick[4] = { p.moveX, p.moveY, p.lookX, p.lookY };
    StickCenter(0, &p.moveX, &p.moveY);
    StickCenter(1, &p.lookX, &p.lookY);
    p.trigL = ReadFloat(g_aTrig, 0); p.trigR = ReadFloat(g_aTrig, 1);
    p.gripL = ReadFloat(g_aGrip, 0); p.gripR = ReadFloat(g_aGrip, 1);
    if (ReadBool(g_aPrimary,   0)) p.buttons |= OLXR_BTN_X;   // left primary  = X
    if (ReadBool(g_aSecondary, 0)) p.buttons |= OLXR_BTN_Y;   // left secondary= Y
    if (ReadBool(g_aPrimary,   1)) p.buttons |= OLXR_BTN_A;   // right primary = A
    if (ReadBool(g_aSecondary, 1)) p.buttons |= OLXR_BTN_B;   // right secondary=B
    if (ReadBool(g_aClick, 0)) p.buttons |= OLXR_BTN_LSTICK;
    if (ReadBool(g_aClick, 1)) p.buttons |= OLXR_BTN_RSTICK;
    if (ReadBool(g_aMenu, 0) || ReadBool(g_aMenu, 1)) p.buttons |= OLXR_BTN_MENU;

    // Both stick clicks recenter from the VR controllers. Swallow the two game buttons while the
    // chord is held so recentering cannot also toggle night vision or trigger a debug binding.
    static bool s_recenterChord = false;
    const bool recenterChord = (p.buttons & OLXR_BTN_LSTICK) && (p.buttons & OLXR_BTN_RSTICK);
    if (recenterChord && !s_recenterChord) {
        OLXR_Recenter();
        XLOG("[OLVR][XRINPUT] both-stick recenter");
    }
    s_recenterChord = recenterChord;
    if (recenterChord) p.buttons &= ~(OLXR_BTN_LSTICK | OLXR_BTN_RSTICK);

    // Hand poses, expressed in the SAME recenter frame the head position uses, so anything built
    // on them later (gestures, hand-relative movement) shares one coordinate system with the lean.
    if (g_fn.locateSpace && g_centered) {
        for (int i = 0; i < 2; ++i) {
            if (!g_handSpace[i]) continue;
            XrSpaceLocation loc = {}; loc.type = XR_TYPE_SPACE_LOCATION_VALUE;
            if (!XrSucceeded(g_fn.locateSpace(g_handSpace[i], g_localSpace, displayTime, &loc))) continue;
            if (!(loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT_VALUE) ||
                !(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT_VALUE)) continue;
            const XrVector3f d = { loc.pose.position.x - g_baseP.x,
                                   loc.pose.position.y - g_baseP.y,
                                   loc.pose.position.z - g_baseP.z };
            const XrVector3f rel = QRotate(QConj(g_baseQ), d);
            const XrQuaternionf q = QMul(QConj(g_baseQ), QNormalize(loc.pose.orientation));
            p.handPose[i].px = rel.x; p.handPose[i].py = rel.y; p.handPose[i].pz = rel.z;
            p.handPose[i].qx = q.x; p.handPose[i].qy = q.y; p.handPose[i].qz = q.z; p.handPose[i].qw = q.w;
            p.handValid |= (1u << i);
        }
        // [LOOKUSE] Aim poses too: "use what your right hand points at" wants the pointing ray,
        // which the grip pose (palm-relative) does not give.
        for (int i = 0; i < 2; ++i) {
            if (!g_aimSpace[i]) continue;
            XrSpaceLocation loc = {}; loc.type = XR_TYPE_SPACE_LOCATION_VALUE;
            if (!XrSucceeded(g_fn.locateSpace(g_aimSpace[i], g_localSpace, displayTime, &loc))) continue;
            if (!(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT_VALUE)) continue;
            const XrQuaternionf q = QMul(QConj(g_baseQ), QNormalize(loc.pose.orientation));
            const XrVector3f rel = QRotate(QConj(g_baseQ), { loc.pose.position.x - g_baseP.x,
                                                              loc.pose.position.y - g_baseP.y,
                                                              loc.pose.position.z - g_baseP.z });
            p.aimPose[i] = { q.x, q.y, q.z, q.w, rel.x, rel.y, rel.z };
            p.aimValid |= (1u << i);
        }
    }

    // [HANDCAM] The right hand's AIM pose, in the recenter frame - the pose the camcorder takes
    // over when the grip is held. Captured here (worker, same display time as everything else)
    // and consumed by the pose block below, which is what publishes to the camera.
    g_handRelValid = 0;
    if (g_fn.locateSpace && g_centered && g_aimSpace[1]) {
        XrSpaceLocation loc = {}; loc.type = XR_TYPE_SPACE_LOCATION_VALUE;
        if (XrSucceeded(g_fn.locateSpace(g_aimSpace[1], g_localSpace, displayTime, &loc)) &&
            (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT_VALUE) &&
            (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT_VALUE)) {
            const XrVector3f d = { loc.pose.position.x - g_baseP.x,
                                   loc.pose.position.y - g_baseP.y,
                                   loc.pose.position.z - g_baseP.z };
            g_handRelP = QRotate(QConj(g_baseQ), d);
            g_handRelQ = QMul(QConj(g_baseQ), QNormalize(loc.pose.orientation));
            g_handRelValid = 1;
        }
    }

    if (g_devFakeHands) {
        DevFakeHands(p.handPose, &p.handValid, displayTime);
        // Aim rays follow the grips (look-to-use with the hand, and anything pointing).
        p.aimPose[0] = p.handPose[0]; p.aimPose[1] = p.handPose[1]; p.aimValid = 3u;
        if (g_devHand[0].grip > p.gripL) p.gripL = g_devHand[0].grip;
        if (g_devHand[1].grip > p.gripR) p.gripR = g_devHand[1].grip;
        if (GetTickCount64() < g_devBtnUntil) p.buttons |= g_devBtnMask;
        if (g_devTrig[0] > p.trigL) p.trigL = g_devTrig[0];
        if (g_devTrig[1] > p.trigR) p.trigR = g_devTrig[1];
        if (GetTickCount64() < g_devStickUntil) { p.moveX = g_devStick[0]; p.moveY = g_devStick[1]; p.lookX = g_devStick[2]; p.lookY = g_devStick[3]; }
    }

    AcquireSRWLockExclusive(&g_pubLock);
    g_padPub = p; g_padPubValid = 1;
    ReleaseSRWLockExclusive(&g_pubLock);
    ApplyHaptics();

    if (!g_inputLoggedBind && (p.handValid || p.buttons || fabsf(p.moveX) > 0.1f ||
                               fabsf(p.moveY) > 0.1f || p.trigL > 0.1f || p.trigR > 0.1f)) {
        g_inputLoggedBind = 1;
        XLOG("[OLVR][XRINPUT] first live controller input seen (hands=%u buttons=0x%X move=%.2f,%.2f look=%.2f,%.2f, thumbs %d %d)",
             p.handValid, p.buttons, rawStick[0], rawStick[1], rawStick[2], rawStick[3], s_stickTouch[0], s_stickTouch[1]);
        LogInteractionProfiles("first input");
    }
    // (Test48) [STICKCENTER] while a stick reads something with no thumb on it (or a controller has no touch
    // sensor and a stick reads over 0.15), say so every 15 s - the numbers a drifting controller needs
    {
        static ULONGLONG s_stickLogMs = 0; static int s_stickLogs = 0;
        const ULONGLONG now = GetTickCount64();
        bool odd = false;
        for (int h = 0; h < 2; ++h) {
            const float ax = rawStick[h * 2], ay = rawStick[h * 2 + 1], m = sqrtf(ax * ax + ay * ay);
            if ((s_stickTouch[h] == 0 && m > 0.10f) || (s_stickTouch[h] < 0 && m > 0.15f)) odd = true;
        }
        if (odd && s_stickLogs < 40 && now - s_stickLogMs > 15000) {
            s_stickLogMs = now; ++s_stickLogs;
            XLOG("[OLVR][STICKCENTER] raw left (%.2f %.2f) right (%.2f %.2f) | thumb on it: left %d right %d (-1 no sensor) | drift %d %d | sent left (%.2f %.2f) right (%.2f %.2f) | dead zone %.2f",
                 rawStick[0], rawStick[1], rawStick[2], rawStick[3], s_stickTouch[0], s_stickTouch[1], s_stickDrift[0] ? 1 : 0, s_stickDrift[1] ? 1 : 0,
                 p.moveX, p.moveY, p.lookX, p.lookY, (float)g_stickDeadzone);
        }
    }
}

extern "C" int OLXR_MotionControlsReady() { return (g_inputReady && g_inputEnabled) ? 1 : 0; }
extern "C" long OLXR_RecenterGeneration() {
    return InterlockedCompareExchange(&g_recenterGeneration, 0, 0);
}
// Game-thread read of the published pad. One lock, whole-struct copy - never field by field, or
// the game gets a stick from one sync and a button from the next.
extern "C" int OLXR_GetPad(OLXrPad* out) {
    if (!out || !g_inputReady || !g_inputEnabled) return 0;
    AcquireSRWLockShared(&g_pubLock);
    const long ok = g_padPubValid;
    if (ok) *out = g_padPub;
    ReleaseSRWLockShared(&g_pubLock);
    return ok ? 1 : 0;
}
// [HANDSYNC] Game-thread read of the frozen per-frame hands. Same whole-struct rule as above.
extern "C" int OLXR_GetFrameHands(OLXrFrameHands* out) {
    if (!out || !g_inputReady || !g_inputEnabled) return 0;
    AcquireSRWLockShared(&g_pubLock);
    const LONG ok = g_liveHandsValid;
    if (ok) *out = g_liveHands;
    ReleaseSRWLockShared(&g_pubLock);
    return ok ? 1 : 0;
}

// [INTERACT] Game-thread read of the frozen head pose this frame renders with (orientation as the
// camera uses it, yaw/pitch, roll only if HeadRoll is on). Position: the RAW head, which is what
// the frame's hands are measured against.
extern "C" int OLXR_GetFrameHead(OLXrPose* out) {
    if (!out) return 0;
    AcquireSRWLockShared(&g_pubLock);
    const long ok = g_olPoseValid;
    const XrQuaternionf q = g_headPoseApp.orientation;
    const float rx = g_liveHands.headRaw[0], ry = g_liveHands.headRaw[1], rz = g_liveHands.headRaw[2];
    const bool raw = g_liveHandsValid != 0;
    const XrVector3f p = g_headPoseApp.position;
    ReleaseSRWLockShared(&g_pubLock);
    if (!ok) return 0;
    out->qx = q.x; out->qy = q.y; out->qz = q.z; out->qw = q.w;
    out->px = raw ? rx : p.x; out->py = raw ? ry : p.y; out->pz = raw ? rz : p.z;
    return 1;
}

void OLXR_Recenter() {
    // Every caller of this function is the PLAYER (key, pad, menu button). The boot baseline
    // sets g_recenterReq directly and never comes through here - that is what keeps the two
    // apart for [HEADPOS].
    g_recenterManual = 1;
    g_recenterReq = 1; g_centered = 0; g_olPoseValid = 0;
    g_olYawRad = g_olPitchRad = g_olRollRad = 0.0f;
    g_olHeadX = g_olHeadY = g_olHeadZ = 0.0f;
    g_smoothInit = false;   // re-seed the smoothing filter from the new baseline, don't drift to it
    g_thValid = 0;   // [THEATER] the screen comes back in front of wherever you now face
    // [HANDSYNC] Hands measured against the old baseline are meaningless now; the arm solver
    // falls back to the live pad for the frame or two until the new baseline's hands promote.
    AcquireSRWLockExclusive(&g_pubLock);
    g_liveHandsValid = 0; g_pendHandsValid = 0;
    ReleaseSRWLockExclusive(&g_pubLock);
}
// [HEADPOS] the runtime's own eye separation, metres (0 until the first locate). ol_ue3 derives
// the position scale from it so a real-world lean covers the distance the stereo makes it look.
extern "C" float OLXR_GetRtEyeSep();
// What the menu's info row reports: the resolution each eye actually gets and the FOV the engine
// actually rendered. Both are measured, never assumed.
extern "C" void OLUE3_GetEyeFov(float* fx, float* fy);
extern "C" float OLUE3_GetHalfEye();
extern "C" float OLUE3_GetWorldToMeters();
extern "C" int OLUE3_StereoMode();
extern "C" void OLProxy_GetGameBackbuffer(UINT* w, UINT* h);
// [WIDEFOV] The headset's own per-eye width, for ol_ue3's FOV widening. Degrees; 0 until the
// first locate has answered.
extern "C" void OLXR_GetHmdFovDeg(float* fx, float* fy) {
    float outX = g_rtFovX;
    float outY = g_rtFovY;
    if (g_rtEyeFovValid && g_w && g_h) {
        // Each runtime eye is asymmetric (Quest/VDXR is roughly -54/+40 and -40/+54 degrees).
        // A centred game camera must cover the furthest edge of BOTH eyes, not merely the width of
        // eye zero. It must also cover the runtime's vertical extent at the source image's aspect;
        // otherwise the compositor quite correctly exposes black at the uncovered edges.
        float maxTanX = 0.0f, maxTanY = 0.0f;
        for (int e = 0; e < 2; ++e) {
            const float x0 = fabsf(tanf(g_rtEyeFov[e].angleLeft));
            const float x1 = fabsf(tanf(g_rtEyeFov[e].angleRight));
            const float y0 = fabsf(tanf(g_rtEyeFov[e].angleUp));
            const float y1 = fabsf(tanf(g_rtEyeFov[e].angleDown));
            if (x0 > maxTanX) maxTanX = x0; if (x1 > maxTanX) maxTanX = x1;
            if (y0 > maxTanY) maxTanY = y0; if (y1 > maxTanY) maxTanY = y1;
        }
        const int mode = OLUE3_StereoMode();
        UINT gameW = 0, gameH = 0;
        OLProxy_GetGameBackbuffer(&gameW, &gameH);
        // g_w is now the 3840-wide private pair.  Its 1920-wide storage eye contains an
        // upscaled replay of a logical 960x1080 UE3 eye, so using g_w here incorrectly widens
        // the camera to about 130x135 degrees.  Preserve the game's logical eye aspect while
        // retaining the private target's higher pixel count.
        const float logicalW = (gameW && mode == 1) ? (float)gameW * 0.5f
                             : (gameW ? (float)gameW
                                      : ((mode == 1) ? (float)g_w * 0.5f : (float)g_w));
        const float logicalH = gameH ? (float)gameH : (float)g_h;
        const float aspect = logicalH > 0.0f ? logicalW / logicalH : 1.0f;
        float coverTanX = maxTanX;
        const float verticalNeed = maxTanY * aspect;
        if (verticalNeed > coverTanX) coverTanX = verticalNeed;
        // Stay clear of the projection singularity while still covering all current consumer HMDs.
        const float maxTan = tanf(140.0f * 0.00872664626f);
        if (coverTanX > maxTan) coverTanX = maxTan;
        outX = 2.0f * atanf(coverTanX);
        outY = 2.0f * atanf(coverTanX / aspect);

        static volatile LONG s_loggedDensityFov = 0;
        if (InterlockedCompareExchange(&s_loggedDensityFov, 1, 0) == 0)
            XLOG("[OLVR][FOVDENSITY] game=%ux%u logicalEyeAspect=%.4f private=%ux%u target=%.1fx%.1fdeg",
                gameW, gameH, aspect, g_w, g_h, outX * 57.2957795f, outY * 57.2957795f);
    }
    if (fx) *fx = outX * 57.2957795f;
    if (fy) *fy = outY * 57.2957795f;
}

// [LENSFIT] The frustum the picture in HALF h of the frame must cover: the runtime eye that half is
// routed to (the Mass Effect model routes by half ^ swap). Tangents; 0 = not now (no runtime
// frustum yet, not fitting to it, another stereo model, or switched off).
extern "C" int OLUE3_GetEyeSwap();
extern "C" int OLXR_LensFitTangents(int half, float* l, float* r, float* u, float* d) {
    if (!g_lensFitOn || !g_rtEyeFovValid || !g_fitFrustum || !g_meModel || OLUE3_StereoMode() != 1) return 0;
    const int eye = (half ^ (OLUE3_GetEyeSwap() ? 1 : 0)) & 1;
    const XrFovf f = g_rtEyeFov[eye];
    const float tl = tanf(f.angleLeft), tr = tanf(f.angleRight), tu = tanf(f.angleUp), td = tanf(f.angleDown);
    if (!(tr - tl > 0.2f && tu - td > 0.2f && tr - tl < 12.0f && tu - td < 12.0f)) return 0;
    if (l) *l = tl; if (r) *r = tr; if (u) *u = tu; if (d) *d = td;
    return 1;
}
extern "C" int  OLXR_GetLensFit() { return g_lensFitOn ? 1 : 0; }
extern "C" void OLXR_SetLensFit(int on) { InterlockedExchange(&g_lensFitOn, on ? 1 : 0); }
extern "C" void OLXR_DevSetFakeHands(int on) { InterlockedExchange(&g_devFakeHands, on ? 1 : 0); }
extern "C" void OLXR_DevSetHeadWave(int on) { InterlockedExchange(&g_devHeadWave, on ? 1 : 0); }
extern "C" void OLXR_DevLook(float yawDeg, float pitchDeg) { g_devLookYaw = yawDeg; g_devLookPitch = pitchDeg; }
// (Beta 1.1) [LINKPOSE] dev: the yaw baseline a recenter would take facing yawDeg off LOCAL forward, and the
// declaration switch - together they reproduce the Quest Link "picture off to one side" on any headset.
extern "C" void OLXR_DevBaseYaw(float yawDeg) {
    const float y = yawDeg * 0.0174532925f;
    g_baseQ = { 0.0f, sinf(y * 0.5f), 0.0f, cosf(y * 0.5f) };
}
extern "C" int OLXR_DevDeclLocal(int v) { g_declLocal = v != 0; return g_declLocal ? 1 : 0; }
extern "C" void OLXR_DevSetLatch(int on) { InterlockedExchange(&g_latchOn, on ? 1 : 0); InterlockedExchange(&g_latchArmed, 1); }

extern "C" void OLXR_GetEyeInfo(int* w, int* h, float* fovX, float* fovY) {
    if (w) *w = (int)g_eyeW;
    if (h) *h = (int)g_eyeH;
    float fx = 0.0f, fy = 0.0f;
    OLUE3_GetEyeFov(&fx, &fy);
    if (fovX) *fovX = fx * 57.2957795f;
    if (fovY) *fovY = fy * 57.2957795f;
}

void OLXR_ToggleProjection() {
    g_olUseProjection = g_olUseProjection ? 0 : 1;
    XLOG("[OLVR][XR] layer = %s", g_olUseProjection ? "PROJECTION" : "QUAD (head-locked)");
}

// ---- quaternion helpers ----
static XrQuaternionf QNormalize(XrQuaternionf q) {
    float l = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (l > 1e-6f) { q.x/=l; q.y/=l; q.z/=l; q.w/=l; } else { q.x=q.y=q.z=0.0f; q.w=1.0f; }
    return q;
}
static XrQuaternionf QConj(XrQuaternionf q) { q.x=-q.x; q.y=-q.y; q.z=-q.z; return q; }
static XrQuaternionf QMul(XrQuaternionf a, XrQuaternionf b) {
    XrQuaternionf r;
    r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return QNormalize(r);
}
// ROTATE A VECTOR. â›” This was built on QMul, and QMul NORMALIZES its result - correct for
// composing two orientations, catastrophic here: q*(v,0)*q' has norm |v|, so forcing it to unit
// length threw away the vector's LENGTH and returned pure direction. Every caller that rotated a
// unit vector (the forward vectors, the eye-right axis) was unharmed and it hid for a month; the
// two callers that rotated a REAL length were not:
//   - the head position ([HEADPOS]) came out as a 1.000 m vector EVERY FRAME. Measured in a
//     2026-08-17 run: 2998 of 3000 frames at exactly 1.000 m while stationary. That is the
//     whole story of five failed lean builds - an apparent hard limit at 1 m, wild shaking
//     after a recenter (near the baseline dp is millimetres of sensor noise, and
//     normalising noise swings a FULL METRE around), and the view snapping to face the
//     character or moving when looking down. No amount of scale or filtering can fix a unit
//     vector. The tuning was never the problem;
//   - the cinema screen was planted at 1 m instead of its intended 1.8 m.
// Rodrigues form below: no quaternion multiply, no normalisation, length preserved exactly.
static XrVector3f QRotate(XrQuaternionf q, XrVector3f v) {
    const float ux = q.x, uy = q.y, uz = q.z, s = q.w;
    const float dot = ux * v.x + uy * v.y + uz * v.z;
    const float uu  = ux * ux + uy * uy + uz * uz;
    const float cx = uy * v.z - uz * v.y;
    const float cy = uz * v.x - ux * v.z;
    const float cz = ux * v.y - uy * v.x;
    XrVector3f out;
    out.x = 2.0f * dot * ux + (s * s - uu) * v.x + 2.0f * s * cx;
    out.y = 2.0f * dot * uy + (s * s - uu) * v.y + 2.0f * s * cy;
    out.z = 2.0f * dot * uz + (s * s - uu) * v.z + 2.0f * s * cz;
    return out;
}
static float ClampRange(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static XrVector3f ClampBodyPose(XrVector3f p) {
    if (!g_bodyAnchor) return p;
    const float side = g_bodySideM > 0.0f ? g_bodySideM : 0.0f;
    const float up = g_bodyUpM > 0.0f ? g_bodyUpM : 0.0f;
    const float down = g_bodyDownM > 0.0f ? g_bodyDownM : 0.0f;
    const float forward = g_bodyForwardM > 0.0f ? g_bodyForwardM : 0.0f;
    const float backward = g_bodyBackwardM > 0.0f ? g_bodyBackwardM : 0.0f;
    p.x = ClampRange(p.x, -side, side);
    p.y = ClampRange(p.y, -down, up);
    p.z = ClampRange(p.z, -forward, backward);
    return p;
}
static float WrapPi(float a) {
    while (a >  3.14159265f) a -= 6.28318531f;
    while (a < -3.14159265f) a += 6.28318531f;
    return a;
}
// Normalized lerp between two quaternions, hemisphere-corrected. Cheaper than slerp and
// indistinguishable at the small per-frame steps a head filter takes.
static XrQuaternionf NlerpQuat(XrQuaternionf a, XrQuaternionf b, float t) {
    float d = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    if (d < 0.0f) { b.x=-b.x; b.y=-b.y; b.z=-b.z; b.w=-b.w; }   // shortest arc
    XrQuaternionf r = { a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t,
                        a.z + (b.z-a.z)*t, a.w + (b.w-a.w)*t };
    return QNormalize(r);
}
// Angle between two orientations, in degrees.
static float QuatAngleDeg(XrQuaternionf a, XrQuaternionf b) {
    float d = fabsf(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w);
    if (d > 1.0f) d = 1.0f;
    return 2.0f * acosf(d) * 57.2957795f;
}
static XrVector3f PoseCenter(const XrView* v) {
    XrVector3f p;
    p.x = (v[0].pose.position.x + v[1].pose.position.x) * 0.5f;
    p.y = (v[0].pose.position.y + v[1].pose.position.y) * 0.5f;
    p.z = (v[0].pose.position.z + v[1].pose.position.z) * 0.5f;
    return p;
}
static bool Resolve(const char* name, void** out) {
    PFN_xrVoidFunction fn = nullptr;
    XrResult r = g_getProc(g_instance, name, &fn);
    if (!XrSucceeded(r) || !fn) { XLOG("[OLVR][XR] resolve FAIL %s r=%d", name, r); return false; }
    *out = (void*)fn; return true;
}
#define RES(name, field) do { if (!Resolve(name, (void**)&g_fn.field)) { g_dead = true; return false; } } while(0)
// [XRINPUT] OPTIONAL resolve: a runtime that does not hand these over loses motion controls and
// nothing else. Using the fatal RES here would mean one missing input entry point takes down a
// stereo mod that has worked for a month.
#define RESOPT(name, field) do { if (!Resolve(name, (void**)&g_fn.field)) g_fn.field = nullptr; } while(0)

static DWORD NowMs() { return GetTickCount(); }

// ---- [PERF] (Test35) where a frame's time goes, logged every 5 s --------------------------------
// "Stutters looking around" has several possible owners: the game missing the headset rate, the
// readback stalling the GPU, or the worker re-showing old frames. One line answers which.
static double PerfNowMs() {
    static LARGE_INTEGER f = {};
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}
extern "C" double OLXR_PerfNowMs() { return PerfNowMs(); }
struct PerfStat {
    double sum = 0.0, mx = 0.0; long n = 0;
    void Add(double v) { sum += v; if (v > mx) mx = v; ++n; }
    double Avg() const { return n ? sum / n : 0.0; }
};
static PerfStat g_pfFrame, g_pfGrab, g_pfCopy, g_pfPace, g_pfPresent;   // game thread only
static volatile long g_devPublish = 1;   // [DEVBRIDGE] "xrpub 0" stops publishing (A/B the cost)
extern "C" void OLXR_PerfNotePresent(double ms) { g_pfPresent.Add(ms); }
extern "C" void OLXR_DevSetPublish(int on) { InterlockedExchange(&g_devPublish, on ? 1 : 0); }
static volatile long g_devUpload = 1;    // [DEVBRIDGE] "xrup 0": the bridge skips its upload (A/B)
extern "C" void OLXR_DevSetUpload(int on) { InterlockedExchange(&g_devUpload, on ? 1 : 0); }
// [DEVBRIDGE] "xrsnap <name>": the bridge saves its upload texture - exactly what goes to the
// headset - as dev\<name>.png, to check the hand-over path delivers the right picture.
static wchar_t       g_devXrSnapPath[MAX_PATH] = { 0 };
static volatile long g_devXrSnapReq = 0;
extern "C" void OLXR_DevSnap(const wchar_t* path) {
    wcsncpy_s(g_devXrSnapPath, path, _TRUNCATE);
    InterlockedExchange(&g_devXrSnapReq, 1);
}
extern "C" double OLXR_PerfNowMs();
static long     g_pfHitch25 = 0, g_pfHitch50 = 0;
static SRWLOCK  g_pfLock = SRWLOCK_INIT;                   // guards the worker's half below
static PerfStat g_pfUpload, g_pfWait;
static long     g_pfXrFrames = 0, g_pfXrNew = 0;
// [XRTIMING] (Test36) where the compositor loop spends its time, and the real display cadence:
// the step between consecutive predicted display times (13.9 = 72 Hz, 11.1 = 90 Hz; a 90 Hz
// runtime fed slower shows 22.2 steps mixed in). Worker thread, under g_pfLock.
static PerfStat g_pfBegin, g_pfEnd, g_pfStep, g_pfLoop;
static volatile long g_heldEyeFrames = 0;                  // [HELDFRAME] repeats not redrawn
static volatile LONG g_displayPeriodUs = 0;                // from xrWaitFrame, 0 = unknown
// [PACING] One game frame per headset frame: with the game free-running above the headset rate
// (the shipped OLEngine.ini caps it at 90, the headset runs 72) every displayed frame advances the
// world by one OR two game frames - an uneven cadence that reads as judder whenever the view
// moves. Before publishing a new frame, wait (up to about one display period) for the worker to
// have taken the previous one. A game slower than the headset never waits.
static bool     g_paceToHeadset = true;
static bool     g_asyncReadback = true;    // [ASYNCREAD] see PublishBackbufferAsync
static bool     g_zeroCopy = true;         // [ZEROCOPY] see PublishZeroCopy (needs the D3D9On12 device)

// Create a D3D11 device on the adapter OpenXR requires (matched by LUID).
static bool CreateD3D11OnAdapter(const LUID& luid) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) { XLOG("[OLVR][XR] CreateDXGIFactory1 failed"); return false; }
    IDXGIAdapter1* chosen = nullptr; IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &a) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.AdapterLuid.LowPart == luid.LowPart && d.AdapterLuid.HighPart == luid.HighPart) {
            chosen = a; char nm[160]; size_t n = 0; wcstombs_s(&n, nm, d.Description, _TRUNCATE);
            XLOG("[OLVR][XR] XR adapter matched: %s", nm);
            break;
        }
        a->Release(); a = nullptr;
    }
    D3D_FEATURE_LEVEL fls[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(chosen, chosen ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   0, fls, _countof(fls), D3D11_SDK_VERSION, &g_d3d11, &got, &g_ctx);
    if (chosen) chosen->Release();
    factory->Release();
    if (FAILED(hr)) { XLOG("[OLVR][XR] D3D11CreateDevice hr=0x%08lX", hr); return false; }
    XLOG("[OLVR][XR] D3D11 bridge device created featureLevel=0x%X", got);
    return true;
}

// One-time OpenXR + D3D11 + swapchain bring-up, sized to the game backbuffer.
static bool Init(UINT w, UINT h) {
    g_w = w; g_h = h;

    // [STEAMVRSCENE] escape hatch for the transparency hunt: [XR] BlackSceneLayer=0 stands the
    // placeholder down entirely (never built, never submitted) so a single ini flip A/Bs the
    // "placeholder changes SteamVR's layer interpretation" suspect without a rebuild.
    // [APILAYER] Virtual Desktop installs an IMPLICIT OpenXR API layer (oculus-compatibility)
    // that injects into every OpenXR process on the machine, including this one on SteamVR,
    // and rewrites calls between this mod and the runtime. This mod is not an OculusXR-plugin
    // app, so the layer has no business in this session on ANY runtime - stand it down for this process
    // only (the env var is per-process; system-wide Virtual Desktop is untouched). Escape hatch:
    // [XR] AllowVdCompatLayer=1 puts it back.
    {
        wchar_t exe[MAX_PATH] = { 0 };
        bool allowVdLayer = false;
        if (GetModuleFileNameW(nullptr, exe, MAX_PATH)) {
            wchar_t* slash = wcsrchr(exe, L'\\');
            if (slash) {
                *(slash + 1) = 0;
                wchar_t ini[MAX_PATH]; swprintf_s(ini, L"%soutlastvr.ini", exe);
                g_blackAllowed = GetPrivateProfileIntW(L"XR", L"BlackSceneLayer", 1, ini) != 0;
                if (!g_blackAllowed) XLOG("[OLVR][XRRUNTIME] black scene layer DISABLED by ini ([XR] BlackSceneLayer=0)");
                allowVdLayer = GetPrivateProfileIntW(L"XR", L"AllowVdCompatLayer", 0, ini) != 0;
                g_useWorker = GetPrivateProfileIntW(L"XR", L"SubmitThread", 1, ini) != 0;
                g_paceToHeadset = GetPrivateProfileIntW(L"XR", L"PaceToHeadset", 1, ini) != 0;
                g_asyncReadback = GetPrivateProfileIntW(L"XR", L"AsyncReadback", 1, ini) != 0;
                g_zeroCopy = GetPrivateProfileIntW(L"XR", L"ZeroCopy", 1, ini) != 0;
                g_latchOn = GetPrivateProfileIntW(L"XR", L"FramePoseLatch", 1, ini) ? 1 : 0;      // [FRAMETAG]
                g_lensFitIni = (long)GetPrivateProfileIntW(L"Render", L"LensFit", 1, ini);
                g_lensFitOn = g_lensFitIni ? 1 : 0;                                              // [LENSFIT]
                g_fakeEyeFov = GetPrivateProfileIntW(L"Debug", L"FakeEyeFov", 0, ini) ? 1 : 0;
                XLOG("[OLVR][TEST36] frame pose latch %s, lens-shaped eyes %s%s", g_latchOn ? "ON" : "OFF",
                     g_lensFitOn ? "ON" : "OFF", g_fakeEyeFov ? " [Debug FakeEyeFov: Quest 3 frusta]" : "");
                // -1 = AUTO: on for runtimes that reproject each view to its own eye point
                // (measured true of Oculus/Quest Link), off for the ones the single-centre-pose
                // model was tuned against (VDXR, SteamVR). 0/1 force it either way.
                g_inputEnabled = GetPrivateProfileIntW(L"Input", L"MotionControls", 1, ini) ? 1 : 0;
                g_hapticsOn = GetPrivateProfileIntW(L"Input", L"Haptics", 1, ini) ? 1 : 0;
                // (Test48) [STICKCENTER]
                g_stickAutoCenter = GetPrivateProfileIntW(L"Input", L"StickAutoCenter", 1, ini) ? 1 : 0;
                { wchar_t dz[32]; GetPrivateProfileStringW(L"Input", L"StickDeadzone", L"0.00", dz, 32, ini);
                  float d = (float)_wtof(dz); if (!(d >= 0.0f)) d = 0.0f; if (d > 0.60f) d = 0.60f; g_stickDeadzone = d;
                  XLOG("[OLVR][STICKCENTER] auto-centre %s, dead zone %.2f", g_stickAutoCenter ? "on" : "off", d); }
                { wchar_t hs[32]; GetPrivateProfileStringW(L"Input", L"HapticStrength", L"1.0", hs, 32, ini);
                  OLXR_SetHapticStrength((float)_wtof(hs)); }
                g_handCamOn = GetPrivateProfileIntW(L"Input", L"HandCamcorder", 0, ini) ? 1 : 0;
                { wchar_t hv[32]; GetPrivateProfileStringW(L"Input", L"HandCamcorderStrength", L"1.0", hv, 32, ini);
                  OLXR_SetHandCamStrength((float)_wtof(hv)); }
                g_bodyAnchor = GetPrivateProfileIntW(L"Stage3", L"BodyAnchor", 1, ini) ? 1 : 0;
                g_bodyFollow = GetPrivateProfileIntW(L"Stage3", L"BodyFollow", 1, ini) ? 1 : 0;
                { wchar_t v[32];
                  GetPrivateProfileStringW(L"Stage3", L"BodySideM", L"0.12", v, 32, ini); g_bodySideM = (float)_wtof(v);
                  GetPrivateProfileStringW(L"Stage3", L"BodyForwardM", L"0.10", v, 32, ini); g_bodyForwardM = (float)_wtof(v);
                  GetPrivateProfileStringW(L"Stage3", L"BodyBackwardM", L"0.08", v, 32, ini); g_bodyBackwardM = (float)_wtof(v);
                  GetPrivateProfileStringW(L"Stage3", L"BodyUpM", L"0.18", v, 32, ini); g_bodyUpM = (float)_wtof(v);
                  GetPrivateProfileStringW(L"Stage3", L"BodyDownM", L"0.35", v, 32, ini); g_bodyDownM = (float)_wtof(v); }
                XLOG("[OLVR][BODY6DOF] anchor=%ld side=%.3f forward=%.3f backward=%.3f up=%.3f down=%.3f handCam=%ld",
                     g_bodyAnchor, g_bodySideM, g_bodyForwardM, g_bodyBackwardM,
                     g_bodyUpM, g_bodyDownM, g_handCamOn);
                g_brandOn = GetPrivateProfileIntW(L"Cinema", L"BrandedBackdrop", 1, ini) ? 1 : 0;
                g_cinemaHeadLocked = GetPrivateProfileIntW(L"Cinema", L"HeadLocked", 0, ini) ? 1 : 0;
                g_thEnv    = GetPrivateProfileIntW(L"Cinema", L"Environment", 1, ini) ? 1 : 0;
                g_thCurved = GetPrivateProfileIntW(L"Cinema", L"Curved", 1, ini) ? 1 : 0;
                { wchar_t v[32];
                  GetPrivateProfileStringW(L"Cinema", L"DistanceM", L"2.6", v, 32, ini); g_cinemaDistanceM = (float)_wtof(v);
                  GetPrivateProfileStringW(L"Cinema", L"WidthDeg", L"62.0", v, 32, ini); g_cinemaWidthDeg = (float)_wtof(v); }
                if (g_cinemaDistanceM < 1.8f) g_cinemaDistanceM = 1.8f;
                if (g_cinemaDistanceM > 4.0f) g_cinemaDistanceM = 4.0f;
                if (g_cinemaWidthDeg < 40.0f) g_cinemaWidthDeg = 40.0f;
                if (g_cinemaWidthDeg > 85.0f) g_cinemaWidthDeg = 85.0f;
                XLOG("[OLVR][CINEMA] branded=%ld headLocked=%ld distance=%.2fm width=%.1fdeg environment=%ld curved=%ld",
                     g_brandOn, g_cinemaHeadLocked, g_cinemaDistanceM, g_cinemaWidthDeg, g_thEnv, g_thCurved);
                g_perEyeMode = (int)GetPrivateProfileIntW(L"XR", L"PerEyePose", (UINT)-1, ini);
                g_declLocal = GetPrivateProfileIntW(L"XR", L"DeclareTruePose", 1, ini) != 0;   // [LINKPOSE]
                g_fitMode    = (int)GetPrivateProfileIntW(L"XR", L"FitRuntimeFrustum", (UINT)-1, ini);
                g_nativeEyeResolution = GetPrivateProfileIntW(L"XR", L"NativeEyeResolution", 1, ini) != 0;
                { wchar_t sv[32];
                  GetPrivateProfileStringW(L"XR", L"Sharpness", L"0.35", sv, 32, ini);
                  g_upscaleSharpness = (float)_wtof(sv);
                  if (!isfinite(g_upscaleSharpness)) g_upscaleSharpness = 0.35f;
                  if (g_upscaleSharpness < 0.0f) g_upscaleSharpness = 0.0f;
                  if (g_upscaleSharpness > 1.0f) g_upscaleSharpness = 1.0f;
                  g_upscaler = GetPrivateProfileIntW(L"XR", L"Upscaler", 1, ini) ? 1 : 0; }
                if (!g_useWorker) XLOG("[OLVR][SUBMITTHREAD] disabled by ini - submits ride the game's Present again");
                // [DEVBRIDGE] Developer testing only: use another OpenXR runtime for THIS process
                // (e.g. SteamVR's null headset) without touching the system-wide active runtime.
                { wchar_t rj[MAX_PATH] = { 0 };
                  GetPrivateProfileStringW(L"Debug", L"XrRuntimeJson", L"", rj, MAX_PATH, ini);
                  if (rj[0] && GetFileAttributesW(rj) != INVALID_FILE_ATTRIBUTES) {
                      SetEnvironmentVariableW(L"XR_RUNTIME_JSON", rj);
                      XLOG("[OLVR][XRRUNTIME] DEV override: this process uses %ls", rj);
                  } }
            }
        }
        if (!allowVdLayer) {
            SetEnvironmentVariableW(L"DISABLE_XR_APILAYER_VIRTUALDESKTOP_OCULUS_COMPATIBILITY", L"1");
            XLOG("[OLVR][APILAYER] Virtual Desktop oculus-compatibility layer disabled for this process");
        } else {
            XLOG("[OLVR][APILAYER] Virtual Desktop oculus-compatibility layer left ENABLED by ini");
        }
    }

    g_loaderDll = LoadLibraryW(L"openxr_loader.dll");
    if (!g_loaderDll) { XLOG("[OLVR][XR] openxr_loader.dll not found (need the x64 one next to OLGame.exe) err=%lu", GetLastError()); g_dead = true; return false; }
    g_getProc = (PFN_xrGetInstanceProcAddr)GetProcAddress(g_loaderDll, "xrGetInstanceProcAddr");
    if (!g_getProc) { XLOG("[OLVR][XR] no xrGetInstanceProcAddr"); g_dead = true; return false; }

    // [APILAYER] Log what is ACTUALLY between this process and the runtime. The registry
    // says what is installed; only the loader knows what it will load after disable variables
    // are honoured. Zero layers is the expected line; anything listed here is rewriting these
    // calls and belongs at the top of the suspect list for any runtime-specific weirdness.
    {
        typedef int32_t (OL_XRAPI *PFN_xrEnumerateApiLayerProperties)(uint32_t, uint32_t*, XrApiLayerProperties*);
        PFN_xrVoidFunction enumFn = nullptr;
        g_getProc(0, "xrEnumerateApiLayerProperties", &enumFn);
        if (enumFn) {
            auto enumLayers = (PFN_xrEnumerateApiLayerProperties)enumFn;
            uint32_t n = 0;
            if (XrSucceeded(enumLayers(0, &n, nullptr))) {
                XLOG("[OLVR][APILAYER] active OpenXR API layers in this process: %u", n);
                if (n > 0 && n < 16) {
                    std::vector<XrApiLayerProperties> props(n);
                    for (auto& p : props) { p = {}; p.type = XR_TYPE_API_LAYER_PROPERTIES_VALUE; }
                    if (XrSucceeded(enumLayers(n, &n, props.data())))
                        for (uint32_t i = 0; i < n; ++i)
                            XLOG("[OLVR][APILAYER]   %s v%u", props[i].layerName, props[i].layerVersion);
                }
            }
        }
    }

    PFN_xrVoidFunction createFn = nullptr;
    if (!XrSucceeded(g_getProc(0, "xrCreateInstance", &createFn)) || !createFn) { XLOG("[OLVR][XR] no xrCreateInstance"); g_dead = true; return false; }
    // [THEATER] Ask for the cylinder layer only if the runtime lists it, so the curved screen can
    // never cost a session: without it the theater uses a flat quad.
    const char* exts[2] = { kD3D11ExtensionName, nullptr };
    uint32_t extCount = 1;
    if (g_thCurved) {
        PFN_xrVoidFunction ef = nullptr;
        if (XrSucceeded(g_getProc(0, "xrEnumerateInstanceExtensionProperties", &ef)) && ef) {
            auto enumExt = (PFN_xrEnumerateInstanceExtensionProperties)ef;
            uint32_t n = 0;
            if (XrSucceeded(enumExt(nullptr, 0, &n, nullptr)) && n) {
                std::vector<XrExtensionProperties> props(n);
                for (auto& p : props) { p = {}; p.type = XR_TYPE_EXTENSION_PROPERTIES_VALUE; }
                if (XrSucceeded(enumExt(nullptr, n, &n, props.data())))
                    for (uint32_t i = 0; i < n; ++i)
                        if (!strcmp(props[i].extensionName, kCylinderExtensionName)) g_cylinderExt = true;
            }
        }
        if (g_cylinderExt) exts[extCount++] = kCylinderExtensionName;
        XLOG("[OLVR][THEATER] cylinder layers %s", g_cylinderExt ? "available - curved screen" : "not offered by this runtime - flat screen");
    }
    XrInstanceCreateInfo ici = {}; ici.type = XR_TYPE_INSTANCE_CREATE_INFO_VALUE;
    strcpy_s(ici.applicationInfo.applicationName, "OutlastVR");
    ici.applicationInfo.applicationVersion = 1;
    strcpy_s(ici.applicationInfo.engineName, "OutlastVR");
    ici.applicationInfo.engineVersion = 1;
    ici.applicationInfo.apiVersion = MakeXrVersion(1, 0, 0);
    ici.enabledExtensionCount = extCount; ici.enabledExtensionNames = exts;
    XrResult r = ((PFN_xrCreateInstance)createFn)(&ici, &g_instance);
    if (!XrSucceeded(r) && extCount > 1) {   // belt and braces: never lose the session to the curve
        XLOG("[OLVR][THEATER] xrCreateInstance r=%d with the cylinder extension - retrying without it", r);
        g_cylinderExt = false; ici.enabledExtensionCount = 1;
        r = ((PFN_xrCreateInstance)createFn)(&ici, &g_instance);
    }
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] xrCreateInstance r=%d (no runtime?)", r); g_dead = true; return false; }
    XLOG("[OLVR][XR] instance created");

    RES("xrDestroyInstance", destroyInstance); RES("xrPollEvent", pollEvent);
    RES("xrGetInstanceProperties", getInstanceProperties);

    // [XRRUNTIME] Which runtime is this? Ported from ME1, where both quirks below were found the
    // hard way. Non-fatal: a failure just leaves both flags false and every path stays as it was.
    {
        XrInstanceProperties props = {}; props.type = XR_TYPE_INSTANCE_PROPERTIES_VALUE;
        if (XrSucceeded(g_fn.getInstanceProperties(g_instance, &props))) {
            props.runtimeName[XR_MAX_RUNTIME_NAME_SIZE_VALUE - 1] = '\0';
            g_isOculusRuntime  = strstr(props.runtimeName, "Oculus") != nullptr ||
                                 strstr(props.runtimeName, "Meta")   != nullptr;
            g_isSteamVrRuntime = strstr(props.runtimeName, "SteamVR") != nullptr;
            XLOG("[OLVR][XRRUNTIME] name='%s' metaFovCrop=%d steamVrFovCrop=%d",
                 props.runtimeName, g_isOculusRuntime ? 1 : 0, g_isSteamVrRuntime ? 1 : 0);
            // Per-eye pose was MEASURED to change nothing on Quest (2026-08-05), so it goes back
            // to off by default everywhere - one variable at a time. The ini can still force it.
            g_perEyePose = (g_perEyeMode > 0);
            // [FITFRUSTUM] auto-on for Oculus: the runtime that ignores the declared FOV.
            g_fitFrustum = (g_fitMode < 0) ? g_isOculusRuntime : (g_fitMode != 0);
            // (Test46e) Quest Link / Air Link (Meta's runtime): with the lens-shaped eyes a tester saw "a screen on
            // each eye, really distorting"; [Render] LensFit=0 fixed it. Off by itself there; LensFit=2 keeps it.
            if (g_isOculusRuntime && g_lensFitIni == 1 && g_lensFitOn) {
                InterlockedExchange(&g_lensFitOn, 0);
                XLOG("[OLVR][LENSFIT] off on Meta's runtime (Quest Link) - the centred render ([Render] LensFit=2 forces it on)");
            }
            XLOG("[OLVR][FITFRUSTUM] %s (%s), perEyePose=%s",
                 g_fitFrustum ? "ON - the picture is placed inside the runtime's own frustum" : "off",
                 g_fitMode < 0 ? "auto, by runtime" : "forced by ini",
                 g_perEyePose ? "on" : "off");
        } else {
            XLOG("[OLVR][XRRUNTIME] could not read the runtime name - both quirk fixes stay off");
        }
    }
    RES("xrGetSystem", getSystem); RES("xrEnumerateViewConfigurationViews", enumerateViewConfigurationViews);
    RES("xrGetD3D11GraphicsRequirementsKHR", getD3D11GraphicsRequirements);
    RES("xrCreateSession", createSession); RES("xrDestroySession", destroySession);
    RES("xrCreateSwapchain", createSwapchain); RES("xrDestroySwapchain", destroySwapchain); RES("xrEnumerateSwapchainImages", enumerateSwapchainImages);
    RES("xrAcquireSwapchainImage", acquireSwapchainImage); RES("xrWaitSwapchainImage", waitSwapchainImage);
    RES("xrReleaseSwapchainImage", releaseSwapchainImage); RES("xrBeginSession", beginSession);
    RES("xrEndSession", endSession); RES("xrWaitFrame", waitFrame);
    RES("xrBeginFrame", beginFrame); RES("xrEndFrame", endFrame);
    RES("xrCreateReferenceSpace", createReferenceSpace); RES("xrDestroySpace", destroySpace);
    RES("xrLocateViews", locateViews);
    // [XRINPUT] core OpenXR 1.0, so these should always resolve - but RESOPT, not RES: a missing
    // one costs motion controls only, never the stereo mod that has worked for a month.
    RESOPT("xrStringToPath", stringToPath); RESOPT("xrCreateActionSet", createActionSet);
    RESOPT("xrDestroyActionSet", destroyActionSet); RESOPT("xrCreateAction", createAction);
    RESOPT("xrDestroyAction", destroyAction);
    RESOPT("xrSuggestInteractionProfileBindings", suggestInteractionProfileBindings);
    RESOPT("xrAttachSessionActionSets", attachSessionActionSets); RESOPT("xrSyncActions", syncActions);
    RESOPT("xrGetActionStateBoolean", getActionStateBoolean); RESOPT("xrGetActionStateFloat", getActionStateFloat);
    RESOPT("xrGetActionStateVector2f", getActionStateVector2f);
    RESOPT("xrCreateActionSpace", createActionSpace); RESOPT("xrLocateSpace", locateSpace);
    RESOPT("xrApplyHapticFeedback", applyHapticFeedback); RESOPT("xrStopHapticFeedback", stopHapticFeedback);
    RESOPT("xrGetCurrentInteractionProfile", getCurrentInteractionProfile); RESOPT("xrPathToString", pathToString);   // (Test48)

    XrSystemGetInfo sgi = {}; sgi.type = XR_TYPE_SYSTEM_GET_INFO_VALUE; sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY_VALUE;
    r = g_fn.getSystem(g_instance, &sgi, &g_system);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] xrGetSystem r=%d (headset off?)", r); g_dead = true; return false; }

    // The runtime's recommended dimensions are the proper final target and do not depend on the
    // desktop/game backbuffer size.
    {
        uint32_t n = 0;
        if (XrSucceeded(g_fn.enumerateViewConfigurationViews(
                g_instance, g_system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_VALUE,
                0, &n, nullptr)) && n >= 2 && n < 16) {
            std::vector<XrViewConfigurationView> vv(n);
            for (auto& v : vv) { v = {}; v.type = XR_TYPE_VIEW_CONFIGURATION_VIEW_VALUE; }
            if (XrSucceeded(g_fn.enumerateViewConfigurationViews(
                    g_instance, g_system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_VALUE,
                    n, &n, vv.data()))) {
                g_runtimeEyeW = vv[0].recommendedImageRectWidth;
                g_runtimeEyeH = vv[0].recommendedImageRectHeight;
                for (uint32_t i = 1; i < n; ++i) {
                    if (vv[i].recommendedImageRectWidth > g_runtimeEyeW) g_runtimeEyeW = vv[i].recommendedImageRectWidth;
                    if (vv[i].recommendedImageRectHeight > g_runtimeEyeH) g_runtimeEyeH = vv[i].recommendedImageRectHeight;
                }
                if (g_runtimeEyeW > 4096) g_runtimeEyeW = 4096;
                if (g_runtimeEyeH > 4096) g_runtimeEyeH = 4096;
                OLUE3_SetHiResEye(g_runtimeEyeW, g_runtimeEyeH);   // [HIRES] each eye at what the headset asks
                XLOG("[OLVR][SHARP] runtime recommends %ux%u per eye; native target=%s sharpness=%.2f",
                     g_runtimeEyeW, g_runtimeEyeH, g_nativeEyeResolution ? "ON" : "off", g_upscaleSharpness);
            }
        }
    }

    XrGraphicsRequirementsD3D11KHR req = {}; req.type = XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR_VALUE;
    r = g_fn.getD3D11GraphicsRequirements(g_instance, g_system, &req);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] getD3D11GraphicsRequirements r=%d", r); g_dead = true; return false; }

    if (!CreateD3D11OnAdapter(req.adapterLuid)) { g_dead = true; return false; }

    XrGraphicsBindingD3D11KHR bind = {}; bind.type = XR_TYPE_GRAPHICS_BINDING_D3D11_KHR_VALUE; bind.device = g_d3d11;
    XrSessionCreateInfo sci = {}; sci.type = XR_TYPE_SESSION_CREATE_INFO_VALUE; sci.next = &bind; sci.systemId = g_system;
    r = g_fn.createSession(g_instance, &sci, &g_session);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] xrCreateSession r=%d", r); g_dead = true; return false; }
    XLOG("[OLVR][XR] session created");

    XrReferenceSpaceCreateInfo rsci = {}; rsci.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO_VALUE;
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW_VALUE; rsci.poseInReferenceSpace = IdentityPose();
    if (!XrSucceeded(g_fn.createReferenceSpace(g_session, &rsci, &g_viewSpace))) { XLOG("[OLVR][XR] createReferenceSpace(view) failed"); g_dead = true; return false; }
    XrReferenceSpaceCreateInfo lsci = rsci; lsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL_VALUE;
    if (!XrSucceeded(g_fn.createReferenceSpace(g_session, &lsci, &g_localSpace))) { XLOG("[OLVR][XR] createReferenceSpace(local) failed"); g_dead = true; return false; }
    XrReferenceSpaceCreateInfo asci = lsci;
    if (!XrSucceeded(g_fn.createReferenceSpace(g_session, &asci, &g_appSpace))) { XLOG("[OLVR][XR] createReferenceSpace(app) failed"); g_dead = true; return false; }

    // [XRINPUT] Build the action set HERE: after the session and its spaces exist, and before any
    // frame runs. xrAttachSessionActionSets is one-shot per session and every action must already
    // exist when it is called, so there is no later opportunity.
    BuildActionSet();

    XrSwapchainCreateInfo scci = {}; scci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    scci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    scci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;   // matches the D3D9 A8R8G8B8 readback bitwise
    scci.sampleCount = 1; scci.width = w; scci.height = h; scci.faceCount = 1; scci.arraySize = 1; scci.mipCount = 1;
    r = g_fn.createSwapchain(g_session, &scci, &g_swapchain);
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] xrCreateSwapchain r=%d %ux%u", r, w, h); g_dead = true; return false; }

    uint32_t imgCount = 0; g_fn.enumerateSwapchainImages(g_swapchain, 0, &imgCount, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> imgs(imgCount);
    for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
    r = g_fn.enumerateSwapchainImages(g_swapchain, imgCount, &imgCount, (XrSwapchainImageBaseHeader*)imgs.data());
    if (!XrSucceeded(r)) { XLOG("[OLVR][XR] enumerateSwapchainImages r=%d", r); g_dead = true; return false; }
    for (auto& im : imgs) g_xrImages.push_back(im.texture);
    XLOG("[OLVR][XR] swapchain %ux%u images=%u", w, h, imgCount);
    if (!g_xrImages.empty() && g_xrImages[0]) {
        D3D11_TEXTURE2D_DESC dd; g_xrImages[0]->GetDesc(&dd);
        XLOG("[OLVR][GAMMAFMT] mono swapchain: asked fmt=%d B8G8R8A8_UNORM_SRGB, runtime allocated fmt=%d%s",
             (int)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, (int)dd.Format,
             dd.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ? " (exact match)" : " (substituted)");
    }

    D3D11_TEXTURE2D_DESC td = {}; td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(g_d3d11->CreateTexture2D(&td, nullptr, &g_uploadTex))) { XLOG("[OLVR][XR] CreateTexture2D(upload) failed"); g_dead = true; return false; }

    XLOG("[OLVR][XR] init OK - waiting for session READY");
    if (g_useWorker) {
        g_workerStop = 0;
        g_workerThread = CreateThread(nullptr, 0, WorkerMain, nullptr, 0, nullptr);
        if (g_workerThread)
            XLOG("[OLVR][SUBMITTHREAD] compositor heartbeat thread running - the game's fps no longer gates the submit cadence");
        else {
            g_useWorker = false;
            XLOG("[OLVR][SUBMITTHREAD] CreateThread FAILED - falling back to inline submits");
        }
    }
    g_inited = true;
    return true;
}

// [XRSTATE] SteamVR transparency forensics: whether the compositor ever promotes this session
// to FOCUSED is the difference between the submitted layers being the scene, and the submitted
// layers being ghosted over the SteamVR environment - and until now only READY/STOPPING were
// ever logged, so that
// question has never had an answer in any log. Name every state, log every transition.
static const char* SessionStateName(int32_t s) {
    switch (s) {
        case 1: return "IDLE";
        case XR_SESSION_STATE_READY_VALUE:        return "READY";
        case XR_SESSION_STATE_SYNCHRONIZED_VALUE: return "SYNCHRONIZED";
        case XR_SESSION_STATE_VISIBLE_VALUE:      return "VISIBLE";
        case XR_SESSION_STATE_FOCUSED_VALUE:      return "FOCUSED";
        case XR_SESSION_STATE_STOPPING_VALUE:     return "STOPPING";
        case XR_SESSION_STATE_LOSS_PENDING_VALUE: return "LOSS_PENDING";
        case XR_SESSION_STATE_EXITING_VALUE:      return "EXITING";
        default: return "?";
    }
}
static volatile LONG g_sessionState = 0;   // last state the runtime reported, for the periodic log
static volatile LONG g_lastEndFrame = 0;   // last xrEndFrame result, for the periodic log
static volatile DWORD g_lastPresentMs = 0; // [EXITGUARD] last time the game presented

static void PollEvents() {
    XrEventDataBuffer ev;
    for (;;) {
        ev = {}; ev.type = XR_TYPE_EVENT_DATA_BUFFER_VALUE;
        XrResult r = g_fn.pollEvent(g_instance, &ev);
        if (r == XR_EVENT_UNAVAILABLE_VALUE || !XrSucceeded(r)) break;
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED_VALUE) {
            auto* ss = reinterpret_cast<XrEventDataSessionStateChanged*>(&ev);
            InterlockedExchange(&g_sessionState, (LONG)ss->state);
            XLOG("[OLVR][XRSTATE] session state -> %d (%s)", ss->state, SessionStateName(ss->state));
            if (ss->state == XR_SESSION_STATE_READY_VALUE) {
                XrSessionBeginInfo sbi = {}; sbi.type = XR_TYPE_SESSION_BEGIN_INFO_VALUE;
                sbi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_VALUE;
                if (XrSucceeded(g_fn.beginSession(g_session, &sbi))) { g_running = true; XLOG("[OLVR][XR] session RUNNING"); }
            } else if (ss->state == XR_SESSION_STATE_STOPPING_VALUE) {
                g_fn.endSession(g_session); g_running = false; XLOG("[OLVR][XR] session STOPPING");
            } else if (ss->state == XR_SESSION_STATE_EXITING_VALUE || ss->state == XR_SESSION_STATE_LOSS_PENDING_VALUE) {
                g_running = false; g_dead = true; XLOG("[OLVR][XR] session EXITING/LOST");
            }
        } else if (ev.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED_VALUE) {
            LogInteractionProfiles("controller layout changed");   // (Test48)
        } else {
            // Anything else the runtime says was previously dropped on the floor. Log each
            // distinct event type once - if SteamVR is trying to signal something about this
            // session (overlay demotion, reference-space change), this is where it would appear.
            static uint32_t s_seenTypes[8] = {};
            static int s_seenCount = 0;
            bool seen = false;
            for (int i = 0; i < s_seenCount; ++i) if (s_seenTypes[i] == ev.type) { seen = true; break; }
            if (!seen && s_seenCount < 8) {
                s_seenTypes[s_seenCount++] = ev.type;
                XLOG("[OLVR][XRSTATE] runtime event type=%u (first occurrence)", ev.type);
            }
        }
    }
}

// Outlast Resets its device after the intro (3840x2160 fullscreen -> 3633x2044 windowed was
// seen live). Everything sized from the backbuffer has to follow, or the copies silently
// mismatch and the headset goes black: UpdateSubresource writes short rows into a wider
// texture, and CopyResource between different-sized textures does nothing at all.
static void DestroyEyeSwapchains() {
    if (g_fn.destroySwapchain) {
        if (g_scL) g_fn.destroySwapchain(g_scL);
        if (g_scR) g_fn.destroySwapchain(g_scR);
    }
    g_scL = g_scR = 0; g_imgL.clear(); g_imgR.clear();
    g_eyeInitTried = false;
}

static bool RebuildForSize(UINT w, UINT h) {
    XLOG("[OLVR][XR] backbuffer changed %ux%u -> %ux%u, rebuilding swapchains", g_w, g_h, w, h);
    DestroyEyeSwapchains();
    if (g_swapchain && g_fn.destroySwapchain) g_fn.destroySwapchain(g_swapchain);
    g_swapchain = 0; g_xrImages.clear();
    if (g_uploadTex) { g_uploadTex->Release(); g_uploadTex = nullptr; }
    g_w = w; g_h = h;

    XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    ci.sampleCount = 1; ci.width = w; ci.height = h; ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
    if (!XrSucceeded(g_fn.createSwapchain(g_session, &ci, &g_swapchain))) {
        XLOG("[OLVR][XR] mono swapchain rebuild FAILED %ux%u", w, h); g_swapchain = 0; return false;
    }
    uint32_t n = 0; g_fn.enumerateSwapchainImages(g_swapchain, 0, &n, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> imgs(n);
    for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
    if (!XrSucceeded(g_fn.enumerateSwapchainImages(g_swapchain, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) return false;
    for (auto& im : imgs) g_xrImages.push_back(im.texture);

    D3D11_TEXTURE2D_DESC td = {}; td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(g_d3d11->CreateTexture2D(&td, nullptr, &g_uploadTex))) {
        XLOG("[OLVR][XR] upload texture rebuild FAILED"); g_uploadTex = nullptr; return false;
    }
    XLOG("[OLVR][XR] rebuild OK: swapchain+upload now %ux%u", w, h);
    return true;
}

// Any D3D9 surface -> SYSTEMMEM -> g_uploadTex. SFR needs this for the saved left-eye RT as
// well as the backbuffer, so the readback is written once against an arbitrary source.
static bool CaptureSurface(IDirect3DDevice9* dev, IDirect3DSurface9* src) {
    if (!src || !g_uploadTex) return false;
    D3DSURFACE_DESC sd;
    if (FAILED(src->GetDesc(&sd))) return false;
    static UINT s_w = 0, s_h = 0; static D3DFORMAT s_fmt = D3DFMT_UNKNOWN;
    if (!g_sysSurf || s_w != sd.Width || s_h != sd.Height || s_fmt != sd.Format) {
        if (g_sysSurf) { g_sysSurf->Release(); g_sysSurf = nullptr; }
        if (SUCCEEDED(dev->CreateOffscreenPlainSurface(sd.Width, sd.Height, sd.Format, D3DPOOL_SYSTEMMEM, &g_sysSurf, nullptr))) {
            s_w = sd.Width; s_h = sd.Height; s_fmt = sd.Format;
            XLOG("[OLVR][XR] readback surface %ux%u fmt=%d", sd.Width, sd.Height, (int)sd.Format);
        } else {
            g_sysSurf = nullptr;
            XLOG("[OLVR][XR] CreateOffscreenPlainSurface FAILED %ux%u fmt=%d", sd.Width, sd.Height, (int)sd.Format);
        }
    }
    if (!g_sysSurf || FAILED(dev->GetRenderTargetData(src, g_sysSurf))) return false;
    D3DLOCKED_RECT lr;
    if (FAILED(g_sysSurf->LockRect(&lr, nullptr, D3DLOCK_READONLY))) return false;
    g_ctx->UpdateSubresource(g_uploadTex, 0, nullptr, lr.pBits, (UINT)lr.Pitch, 0);
    g_sysSurf->UnlockRect();
    return true;
}

static bool CaptureBackbuffer(IDirect3DDevice9* dev) {
    IDirect3DSurface9* frame = OLProxy_AcquireNativeStereo();
    if (!frame && (FAILED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &frame)) || !frame))
        return false;
    bool ok = CaptureSurface(dev, frame);
    frame->Release();
    return ok;
}

// [SUBMITTHREAD] Game-thread half of the bridge: D3D9 readback into a CPU buffer plus the pose
// this frame was rendered with. No D3D11, no OpenXR - those belong to the worker alone. The
// pose rides WITH the pixels so the worker can tag a held frame with the pose of the image it
// is actually showing; tagging a stale image with a fresh pose is the documented cause of
// noticeably shaky head tracking (an earlier attempt).
//
// [ASYNCREAD] (Test35) GetRenderTargetData only QUEUES the copy; the LockRect right behind it
// then waited for the GPU to finish the whole frame and the copy. Measured on the SteamVR null
// headset at 3456x1824: 27 ms average, 48 ms worst, every frame - the game fell to ~20 fps in
// the headset while holding 90 on the monitor. That wait also killed all CPU/GPU overlap. The
// stutter the player saw looking around was this.
// Now the copy goes into one of a small ring of surfaces and is picked up on a LATER Present,
// when the GPU has long finished it (D3DLOCK_DONOTWAIT: never stall for it). The pose and FOV
// are latched into the slot when its copy is queued - the moment that frame is complete - so
// the image still carries exactly the pose it was rendered with; the compositor reprojects the
// extra frame of age like any other. Only if every slot is still in flight does a lock wait.
// [XR] AsyncReadback=0 restores the old same-frame readback (g_asyncReadback, declared above).
// Why not keep the frame on the GPU altogether (a render target shared with the D3D11 bridge)?
// Tried 2026-10-13: the plain D3D9 device refuses shared resources (D3DERR_INVALIDCALL), and
// creating it through the 9Ex runtime crashes Outlast at startup - UE3 makes its textures in
// D3DPOOL_MANAGED, which 9Ex rejects even through the ordinary CreateDevice.
// [ASYNCREAD] Copying straight out of the swap chain's backbuffer still serialized the GPU: the
// backbuffer is reused next frame, so the driver finished everything before letting the copy
// run. Measured in gameplay on the null headset: 71 fps with publishing off, 24 with it on, the
// GPU going LESS busy (94% -> 71%) - waiting, not working. So each slot first takes a plain GPU
// copy into its own render target (StretchRect, VRAM to VRAM, no sync), and the copy to system
// memory reads from THAT, fenced by an event query; the lock happens only once the query says it
// is done. The same scheme screen-capture tools use on D3D9.
struct ReadSlot {
    IDirect3DSurface9* surf = nullptr;   // SYSTEMMEM, what the CPU reads
    IDirect3DSurface9* rt = nullptr;     // DEFAULT pool staging copy of the backbuffer
    IDirect3DQuery9*   q = nullptr;      // signals when rt -> surf has landed
    bool     pending = false;
    uint64_t order = 0;
    XrPosef  pose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
    long     poseValid = 0;
    float    fovX = 0.0f, fovY = 0.0f;
    long     lensFit = 0;
};
static const int kReadSlots = 3;
static ReadSlot  g_rs[kReadSlots];
static UINT      g_rsW = 0, g_rsH = 0;
static D3DFORMAT g_rsFmt = D3DFMT_UNKNOWN;
static uint64_t  g_rsOrder = 0;

static void ReadRingRelease() {
    for (auto& s : g_rs) {
        if (s.q) s.q->Release();
        if (s.rt) s.rt->Release();
        if (s.surf) s.surf->Release();
        s = ReadSlot();
    }
    g_rsW = g_rsH = 0; g_rsFmt = D3DFMT_UNKNOWN;
}
// The staging render targets are DEFAULT pool: they must be gone before the device is Reset.
static void ZcReleaseFrames();
static void ExReleaseFrames();   // [EXSHARE]
extern "C" void OLXR_OnDeviceLost() { ReadRingRelease(); ZcReleaseFrames(); ExReleaseFrames(); }
static bool ReadRingEnsure(IDirect3DDevice9* dev, const D3DSURFACE_DESC& sd) {
    if (g_rs[0].surf && g_rsW == sd.Width && g_rsH == sd.Height && g_rsFmt == sd.Format) return true;
    ReadRingRelease();
    for (auto& s : g_rs) {
        if (FAILED(dev->CreateOffscreenPlainSurface(sd.Width, sd.Height, sd.Format, D3DPOOL_SYSTEMMEM, &s.surf, nullptr)) || !s.surf) {
            XLOG("[OLVR][ASYNCREAD] readback surface %ux%u FAILED - falling back to same-frame readback", sd.Width, sd.Height);
            ReadRingRelease();
            return false;
        }
    }
    int staged = 0;
    for (auto& s : g_rs) {
        if (SUCCEEDED(dev->CreateRenderTarget(sd.Width, sd.Height, sd.Format, D3DMULTISAMPLE_NONE, 0, FALSE, &s.rt, nullptr)) && s.rt &&
            SUCCEEDED(dev->CreateQuery(D3DQUERYTYPE_EVENT, &s.q)) && s.q) { ++staged; continue; }
        if (s.q) { s.q->Release(); s.q = nullptr; }
        if (s.rt) { s.rt->Release(); s.rt = nullptr; }
    }
    g_rsW = sd.Width; g_rsH = sd.Height; g_rsFmt = sd.Format;
    XLOG("[OLVR][ASYNCREAD] %d readback surfaces %ux%u fmt=%d, %d with a GPU staging copy + fence",
         kReadSlots, sd.Width, sd.Height, (int)sd.Format, staged);
    return true;
}

// Rows of a locked surface into the published buffer, with the tag the caller supplies. Caller
// holds g_pubLock exclusively.
static void CopyToPublished(const D3DLOCKED_RECT& lr, UINT w, UINT h) {
    const UINT rowBytes = w * 4;
    g_pubPixels.resize((size_t)rowBytes * h);
    const uint8_t* src = (const uint8_t*)lr.pBits;
    if ((UINT)lr.Pitch == rowBytes) memcpy(g_pubPixels.data(), src, (size_t)rowBytes * h);
    else for (UINT y = 0; y < h; ++y)
        memcpy(g_pubPixels.data() + (size_t)y * rowBytes, src + (size_t)y * lr.Pitch, rowBytes);
    g_pubW = w; g_pubH = h; g_pubPitch = rowBytes;
}

// [POSEFREEZE] The frame just finished: promote the pending pose so the NEXT frame renders frozen
// at the newest locate. Caller holds g_pubLock exclusively, and has already latched this frame's
// own tag (promote after tagging, or the image carries a pose one frame newer than its pixels).
static void PromotePendingPose() {
    if (!g_pendValid) return;
    g_headPoseApp = g_pendPose;
    g_olYawRad = g_pendYaw; g_olPitchRad = g_pendPitch; g_olRollRad = g_pendRoll;
    g_olHeadX = g_pendPose.position.x; g_olHeadY = g_pendPose.position.y;
    g_olHeadZ = g_pendPose.position.z;
    g_olPoseValid = 1;
    if (g_pendHandsValid) { g_liveHands = g_pendHands; g_liveHandsValid = 1; }
    g_rwLat[0] = g_rwPend[0]; g_rwLat[1] = g_rwPend[1];   // [ROOMWALK]
}

extern "C" int OLUE3_LensFitLive();
extern "C" int OLInteract_Get(int which);   // (Test46) 12 = [Motion] ViewfinderFrame
static bool FrameLatchActive() { return g_latchOn && g_useWorker && g_latchLive > 0; }

// [FRAMETAG] GAME thread. Every reader of the per-frame pose calls this first; the first call
// after a frame's views were built promotes the newest locate, the rest of the frame is frozen.
extern "C" void OLXR_FrameLatch() {
    if (!g_latchOn || !g_useWorker) return;
    if (!InterlockedExchange(&g_latchArmed, 0)) return;
    AcquireSRWLockExclusive(&g_pubLock);
    PromotePendingPose();
    ReleaseSRWLockExclusive(&g_pubLock);
}

// [ROOMWALK] (Test44) GAME thread: the room steps (m, recenter x right / z back) the frame's pose has
// taken since the last call - the pose latched for this frame, or the live one without the latch.
extern "C" int OLXR_RoomWalkTake(float out[2]) {
    AcquireSRWLockShared(&g_pubLock);
    const double x = g_rwLat[0], z = g_rwLat[1];
    ReleaseSRWLockShared(&g_pubLock);
    out[0] = (float)(x - g_rwDone[0]); out[1] = (float)(z - g_rwDone[1]);
    g_rwDone[0] = x; g_rwDone[1] = z;
    return (out[0] != 0.0f || out[1] != 0.0f) ? 1 : 0;
}

// [FRAMETAG] GAME thread, after both eye views of a frame are built: record what they used, keyed
// by the left view's view+projection matrices (FSceneView+0x80, 128 bytes).
extern "C" float OLInteract_ViewfinderZoom();
// lensFit carries bit 0 = lens-shaped eyes, bit 1 = [VIEWFINDER] camcorder zoom frame.
extern "C" void OLXR_FrameCommit(float fovX, float fovY, int lensFit, int viewfinder, const float* key32) {
    g_tidGame = GetCurrentThreadId();
    if (!g_latchOn || !g_useWorker || !key32) return;
    AcquireSRWLockExclusive(&g_pubLock);
    FrameTag& t = g_tagRing[g_tagNext & 7];
    ++g_tagNext;
    t.pose = g_headPoseApp; t.poseValid = g_olPoseValid;
    t.fovX = fovX; t.fovY = fovY; t.lensFit = (lensFit ? 1 : 0) | (viewfinder ? 2 : 0);
    memcpy(t.key, key32, sizeof(t.key));
    ReleaseSRWLockExclusive(&g_pubLock);
    InterlockedExchange(&g_latchLive, 8);
    InterlockedExchange(&g_latchArmed, 1);
}

// [FRAMETAG] RENDER thread, from the scene renderer's per-view setup (ol_ue3's [PVS] hook, called
// from InitViews with the render thread's copy of each view): which committed frame is being
// drawn right now. The copy carries the game thread's matrices bit for bit, so the lookup is an
// exact compare and cannot pick a neighbouring frame. Two frames with identical matrices have
// identical camera poses, so either answer is right.
static FrameTag      g_rtTag;
static volatile long g_rtTagFresh = 0;
extern "C" void OLXR_RenderViewSeen(const void* key128) {
    if (!g_latchOn || !g_useWorker || !key128) return;
    AcquireSRWLockShared(&g_pubLock);
    bool hit = false;
    for (long i = 1; i <= 8 && i <= g_tagNext; ++i) {
        const FrameTag& t = g_tagRing[(g_tagNext - i) & 7];
        if (memcmp(t.key, key128, sizeof(t.key)) == 0) { g_rtTag = t; hit = true; break; }
    }
    ReleaseSRWLockShared(&g_pubLock);
    if (hit) { g_rtTagFresh = 1; InterlockedIncrement(&g_tagStat[3]); }
    else InterlockedIncrement(&g_tagStat[2]);
}

// [FRAMETAG] PRESENT (render thread), once per Present whether or not the frame is published: the
// tag of the image this Present finishes - the frame the renderer identified while drawing it.
// Without one (menus, the theater screen, a level with no view state, the latch switched off)
// the live values are the tag, as before.
static FrameTag g_curTag = { { { 0, 0, 0, 1 }, { 0, 0, 0 } }, 0, 0.0f, 0.0f, 0 };
static void PresentPopTag() {
    g_tidPresent = GetCurrentThreadId();
    if (g_latchLive > 0) InterlockedDecrement(&g_latchLive);
    AcquireSRWLockExclusive(&g_pubLock);
    const XrPosef live = g_headPoseApp;
    if (g_latchOn && g_useWorker && g_rtTagFresh) {
        g_curTag = g_rtTag;
        InterlockedIncrement(&g_tagStat[0]);
        if (memcmp(&g_curTag.pose, &live, sizeof(live)) != 0) InterlockedIncrement(&g_tagDiff);
    } else {
        if (FrameLatchActive()) InterlockedIncrement(&g_tagStat[1]);
        g_curTag.pose = live; g_curTag.poseValid = g_olPoseValid;
        OLUE3_GetEyeFov(&g_curTag.fovX, &g_curTag.fovY);
        g_curTag.lensFit = OLUE3_LensFitLive() | (OLInteract_ViewfinderZoom() > 0.0f ? 2 : 0);
    }
    g_rtTagFresh = 0;
    ReleaseSRWLockExclusive(&g_pubLock);
}
// g_pubLock held: the tag PresentPopTag chose for this Present.
static void TakeFrameTag(XrPosef* pose, LONG* valid, float* fx, float* fy, long* lensFit) {
    *pose = g_curTag.pose; *valid = g_curTag.poseValid;
    *fx = g_curTag.fovX; *fy = g_curTag.fovY; *lensFit = g_curTag.lensFit;
}

// The Test35 Present-time promote, kept only for frames the game thread does not latch.
static void PromoteAtPresent() {
    if (!FrameLatchActive()) PromotePendingPose();
}

// Lock one pending slot and publish it. wait=false: give up (return false) if the GPU has not
// finished it yet.
static bool ReadSlotCollect(ReadSlot& s, bool wait) {
    if (s.q) {
        HRESULT qr;
        while ((qr = s.q->GetData(nullptr, 0, D3DGETDATA_FLUSH)) == S_FALSE) {
            if (!wait) return false;
            Sleep(0);
        }
        if (FAILED(qr)) { s.pending = false; return false; }   // device lost: drop the frame
    }
    D3DLOCKED_RECT lr;
    const HRESULT hr = s.surf->LockRect(&lr, nullptr, D3DLOCK_READONLY | (wait ? 0 : D3DLOCK_DONOTWAIT));
    if (hr == D3DERR_WASSTILLDRAWING) return false;
    s.pending = false;
    if (FAILED(hr)) return false;
    AcquireSRWLockExclusive(&g_pubLock);
    CopyToPublished(lr, g_rsW, g_rsH);
    g_pubZcSlot = -1; g_pubExSlot = -1;
    g_pubPose = s.pose; g_pubPoseValid = s.poseValid;
    g_pubFovX = s.fovX; g_pubFovY = s.fovY;   // [FOVLATCH] the angle THESE pixels cover
    g_pubLensFit = s.lensFit;
    ++g_pubSeq;
    ReleaseSRWLockExclusive(&g_pubLock);
    s.surf->UnlockRect();
    return true;
}

static bool PublishBackbufferAsync(IDirect3DDevice9* dev, IDirect3DSurface9* bb, const D3DSURFACE_DESC& sd) {
    if (!ReadRingEnsure(dev, sd)) return false;
    const double t0 = PerfNowMs();
    // After a gap (session paused, loading hitch) the copies still in flight are old pictures:
    // drop them rather than flash a stale frame on resume.
    static double s_lastCall = 0.0;
    if (s_lastCall > 0.0 && t0 - s_lastCall > 150.0) for (auto& s : g_rs) s.pending = false;
    s_lastCall = t0;
    // 1. Collect finished copies, oldest first. Publishing an older one after a newer one would
    //    step backwards in time, so anything older than a collected slot is simply dropped.
    bool published = false;
    for (;;) {
        ReadSlot* oldest = nullptr;
        for (auto& s : g_rs) if (s.pending && (!oldest || s.order < oldest->order)) oldest = &s;
        if (!oldest) break;
        if (!ReadSlotCollect(*oldest, false)) break;   // still in flight: so are the newer ones
        published = true;
    }
    // 2. A free slot for this frame; if all are in flight, wait for the oldest.
    ReadSlot* slot = nullptr;
    for (auto& s : g_rs) if (!s.pending) { slot = &s; break; }
    if (!slot) {
        ReadSlot* oldest = nullptr;
        for (auto& s : g_rs) if (!oldest || s.order < oldest->order) oldest = &s;
        if (ReadSlotCollect(*oldest, true)) published = true;
        slot = oldest;
    }
    const double t1 = PerfNowMs();
    g_pfCopy.Add(t1 - t0);
    // 3. Queue this frame's copy with the pose it was rendered with, then move the live pose on.
    if (slot->rt) {
        if (FAILED(dev->StretchRect(bb, nullptr, slot->rt, nullptr, D3DTEXF_NONE)) ||
            FAILED(dev->GetRenderTargetData(slot->rt, slot->surf))) return published;
        slot->q->Issue(D3DISSUE_END);
    } else if (FAILED(dev->GetRenderTargetData(bb, slot->surf))) return published;
    g_pfGrab.Add(PerfNowMs() - t1);
    slot->pending = true; slot->order = ++g_rsOrder;
    AcquireSRWLockExclusive(&g_pubLock);
    TakeFrameTag(&slot->pose, &slot->poseValid, &slot->fovX, &slot->fovY, &slot->lensFit);
    PromoteAtPresent();
    ReleaseSRWLockExclusive(&g_pubLock);
    return true;
}

// ---- [ZEROCOPY] (Test35) the headset frame never leaves the GPU ---------------------------------
// Every D3D9 readback is a full CPU/GPU sync on this driver: measured in gameplay on the null
// headset, 71 fps with publishing paused and 24 with it on, whatever the readback's size. With the
// game running on Windows' D3D9On12 layer ([XR] D3D9On12=1, d3d9_proxy.cpp) each D3D9 resource has
// a D3D12 resource underneath, and D3D9On12 lends it out with GPU-side fencing. So:
//   game:   StretchRect backbuffer -> our D3D9 render target (GPU copy)
//           unwrap it, D3D12 copy queue: -> a texture SHARED with the bridge, signal fence F = n
//           hand the D3D9 target back, fenced on F = n (D3D9On12 waits for our copy on the GPU)
//   bridge: GPU-wait F >= n, CopyResource into the upload texture, signal fence R = n
// The CPU never waits on the GPU anywhere, and the frame is published the moment it is queued, so
// it reaches the headset a frame sooner than the readback ring could deliver it.
// Slots: 0 free, 2 published (bridge has not taken it), 3 bridge copying. A slot is reused only
// when the D3D12 copy into it is done (F) AND, if the bridge took it, the bridge's copy out of it
// is done (R). Any failure drops to the readback path for the rest of the session, logged once.
static const int kZcSlots = 4;
struct ZcSlot {
    IDirect3DTexture9*         tex9 = nullptr;
    IDirect3DSurface9*         surf9 = nullptr;
    ID3D12Resource*            shared12 = nullptr;
    HANDLE                     sharedH = nullptr;
    ID3D12CommandAllocator*    alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    volatile long              state = 0;
    uint64_t                   value = 0;      // F value of the copy into it
    bool                       taken = false;  // the bridge read it: wait for R >= value too
    XrPosef                    pose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
    long                       poseValid = 0;
    float                      fovX = 0.0f, fovY = 0.0f;
    long                       lensFit = 0;
    ID3D11Texture2D*           tex11 = nullptr;  // bridge-owned
    HANDLE                     opened = nullptr; // bridge-owned
};
static ZcSlot   g_zc[kZcSlots];
static volatile long g_zcState = 0;           // 0 untried, 1 in use, -1 unavailable this session
static IDirect3DDevice9On12* g_zc9 = nullptr;
static ID3D12Device*        g_zc12 = nullptr;
static ID3D12CommandQueue*  g_zcQueue = nullptr;
static ID3D12Fence*         g_zcF = nullptr;   // frames copied (game's copy queue signals)
static ID3D12Fence*         g_zcR = nullptr;   // frames read (the bridge signals)
static HANDLE               g_zcFH = nullptr, g_zcRH = nullptr;
static uint64_t             g_zcNext = 0;
static UINT                 g_zcW = 0, g_zcH = 0;
static volatile long        g_zcGen = 0;       // bumped when the game side is rebuilt
static long                 g_zcSkipped = 0;   // frames not published: every slot busy
// bridge-owned
static ID3D11Device5*        g_zc11 = nullptr;
static ID3D11DeviceContext4* g_zcCtx = nullptr;
static ID3D11Fence*          g_zcF11 = nullptr;
static ID3D11Fence*          g_zcR11 = nullptr;
static long                  g_zcWorkerGen = -1;

static void ZcFail(const char* why, long hr) {
    if (InterlockedExchange(&g_zcState, -1) != -1)
        XLOG("[OLVR][ZEROCOPY] unavailable (%s, hr=0x%08lX) - frames go through the readback path", why, hr);
}
template <class T> static void ZcRel(T*& p) { if (p) { p->Release(); p = nullptr; } }

// Game thread. Waits (CPU) for our own outstanding D3D12 copies, then drops every game-side
// object that depends on the frame size. Shared fences and the queue survive.
static void ZcReleaseFrames() {
    if (g_zcF && g_zcNext && g_zcF->GetCompletedValue() < g_zcNext) {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (ev && SUCCEEDED(g_zcF->SetEventOnCompletion(g_zcNext, ev))) WaitForSingleObject(ev, 500);
        if (ev) CloseHandle(ev);
    }
    AcquireSRWLockExclusive(&g_pubLock);
    InterlockedIncrement(&g_zcGen);
    for (auto& s : g_zc) {
        ZcRel(s.surf9); ZcRel(s.tex9); ZcRel(s.list); ZcRel(s.alloc); ZcRel(s.shared12);
        if (s.sharedH) { CloseHandle(s.sharedH); s.sharedH = nullptr; }
        InterlockedExchange(&s.state, 0); s.value = 0; s.taken = false;
    }
    if (g_pubZcSlot >= 0) { g_pubZcSlot = -1; g_pubPixels.clear(); }
    g_zcW = g_zcH = 0;
    ReleaseSRWLockExclusive(&g_pubLock);
}

// Game thread, first use: is this a D3D9On12 device on the same GPU as the bridge?
static bool ZcInit(IDirect3DDevice9* dev) {
    if (g_zcState < 0) return false;
    if (g_zcQueue) return true;
    HRESULT hr = dev->QueryInterface(__uuidof(IDirect3DDevice9On12), (void**)&g_zc9);
    if (FAILED(hr) || !g_zc9) { g_zc9 = nullptr; ZcFail("the game's device is not D3D9On12 ([XR] D3D9On12=0?)", hr); return false; }
    if (FAILED(hr = g_zc9->GetD3D12Device(__uuidof(ID3D12Device), (void**)&g_zc12)) || !g_zc12) { ZcFail("GetD3D12Device", hr); return false; }
    {   // the bridge's D3D11 device must sit on the same adapter
        LUID l12 = g_zc12->GetAdapterLuid(), l11 = {};
        IDXGIDevice* dx = nullptr; IDXGIAdapter* ad = nullptr; DXGI_ADAPTER_DESC ds = {};
        if (g_d3d11 && SUCCEEDED(g_d3d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&dx)) &&
            SUCCEEDED(dx->GetAdapter(&ad)) && SUCCEEDED(ad->GetDesc(&ds))) l11 = ds.AdapterLuid;
        ZcRel(ad); ZcRel(dx);
        if (l11.LowPart != l12.LowPart || l11.HighPart != l12.HighPart) { ZcFail("the game and the headset bridge are on different GPUs", 0); return false; }
    }
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    if (FAILED(hr = g_zc12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_zcQueue))) { g_zcQueue = nullptr; ZcFail("copy queue", hr); return false; }
    if (FAILED(hr = g_zc12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&g_zcF)) ||
        FAILED(hr = g_zc12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&g_zcR)) ||
        FAILED(hr = g_zc12->CreateSharedHandle(g_zcF, nullptr, GENERIC_ALL, nullptr, &g_zcFH)) ||
        FAILED(hr = g_zc12->CreateSharedHandle(g_zcR, nullptr, GENERIC_ALL, nullptr, &g_zcRH))) {
        ZcFail("shared fences", hr); return false;
    }
    XLOG("[OLVR][ZEROCOPY] D3D9On12 device on the bridge's GPU - frames will be handed over on the GPU");
    return true;
}

// Game thread: the per-size objects. The shared texture copies the format of the unwrapped
// D3D9 target, so its D3D12 twin is unwrapped once here to read it.
static bool ZcEnsureFrames(IDirect3DDevice9* dev, UINT w, UINT h) {
    if (g_zc[0].surf9 && g_zcW == w && g_zcH == h) return true;
    ZcReleaseFrames();
    HRESULT hr;
    for (auto& s : g_zc) {
        if (FAILED(hr = dev->CreateTexture(w, h, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &s.tex9, nullptr)) ||
            FAILED(hr = s.tex9->GetSurfaceLevel(0, &s.surf9))) { ZcReleaseFrames(); ZcFail("D3D9 frame target", hr); return false; }
        ID3D12Resource* r = nullptr;
        if (FAILED(hr = g_zc9->UnwrapUnderlyingResource(s.tex9, g_zcQueue, __uuidof(ID3D12Resource), (void**)&r)) || !r) {
            ZcReleaseFrames(); ZcFail("UnwrapUnderlyingResource", hr); return false;
        }
        D3D12_RESOURCE_DESC rd = r->GetDesc();
        r->Release();
        UINT64 zero = 0;
        g_zc9->ReturnUnderlyingResource(s.tex9, 0, &zero, nullptr);
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC sd = {};
        sd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; sd.Width = w; sd.Height = h; sd.DepthOrArraySize = 1;
        sd.MipLevels = 1; sd.Format = rd.Format; sd.SampleDesc.Count = 1; sd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (FAILED(hr = g_zc12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &sd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        __uuidof(ID3D12Resource), (void**)&s.shared12)) ||
            FAILED(hr = g_zc12->CreateSharedHandle(s.shared12, nullptr, GENERIC_ALL, nullptr, &s.sharedH)) ||
            FAILED(hr = g_zc12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, __uuidof(ID3D12CommandAllocator), (void**)&s.alloc)) ||
            FAILED(hr = g_zc12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, s.alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&s.list))) {
            ZcReleaseFrames(); ZcFail("shared frame texture", hr); return false;
        }
        s.list->Close();
    }
    g_zcW = w; g_zcH = h;
    XLOG("[OLVR][ZEROCOPY] %d shared frames %ux%u", kZcSlots, w, h);
    return true;
}

// Game thread, once per Present.
static bool PublishZeroCopy(IDirect3DDevice9* dev, IDirect3DSurface9* bb) {
    const double t0 = PerfNowMs();
    const uint64_t doneF = g_zcF->GetCompletedValue(), doneR = g_zcR->GetCompletedValue();
    // Free slots the bridge has finished reading.
    for (auto& s : g_zc)
        if (s.state == 3 && doneR >= s.value) { s.taken = true; InterlockedCompareExchange(&s.state, 0, 3); }
    ZcSlot* slot = nullptr;
    for (auto& s : g_zc)
        if (s.state == 0 && doneF >= s.value && (!s.taken || doneR >= s.value)) { slot = &s; break; }
    bool published = false;
    if (slot && SUCCEEDED(dev->StretchRect(bb, nullptr, slot->surf9, nullptr, D3DTEXF_NONE))) {
        ID3D12Resource* src = nullptr;
        HRESULT hr = g_zc9->UnwrapUnderlyingResource(slot->tex9, g_zcQueue, __uuidof(ID3D12Resource), (void**)&src);
        if (SUCCEEDED(hr) && src) {
            slot->alloc->Reset();
            slot->list->Reset(slot->alloc, nullptr);
            slot->list->CopyResource(slot->shared12, src);
            slot->list->Close();
            ID3D12CommandList* lists[] = { slot->list };
            g_zcQueue->ExecuteCommandLists(1, lists);
            const uint64_t v = ++g_zcNext;
            g_zcQueue->Signal(g_zcF, v);
            src->Release();
            ID3D12Fence* f = g_zcF;
            g_zc9->ReturnUnderlyingResource(slot->tex9, 1, (UINT64*)&v, &f);
            slot->value = v; slot->taken = false;
            AcquireSRWLockExclusive(&g_pubLock);
            // A published frame the bridge never took is superseded: free it.
            if (g_pubZcSlot >= 0 && g_pubZcSlot < kZcSlots) InterlockedCompareExchange(&g_zc[g_pubZcSlot].state, 0, 2);
            TakeFrameTag(&slot->pose, &slot->poseValid, &slot->fovX, &slot->fovY, &slot->lensFit);
            InterlockedExchange(&slot->state, 2);
            g_pubZcSlot = (long)(slot - g_zc); g_pubExSlot = -1;
            g_pubW = g_zcW; g_pubH = g_zcH; g_pubPitch = g_zcW * 4;
            g_pubPose = slot->pose; g_pubPoseValid = slot->poseValid;
            g_pubFovX = slot->fovX; g_pubFovY = slot->fovY;   // [FOVLATCH]
            g_pubLensFit = slot->lensFit;
            ++g_pubSeq;
            ReleaseSRWLockExclusive(&g_pubLock);
            published = true;
            if (g_zcState == 0) InterlockedExchange(&g_zcState, 1);
        } else {
            ZcFail("UnwrapUnderlyingResource (per frame)", hr);
        }
    } else if (!slot) {
        ++g_zcSkipped;
    }
    g_pfCopy.Add(PerfNowMs() - t0);
    AcquireSRWLockExclusive(&g_pubLock);
    PromoteAtPresent();   // always: a frame that could not be sent must not hold the next one back
    ReleaseSRWLockExclusive(&g_pubLock);
    return published;
}

// Bridge thread: drop textures opened for an older generation of shared frames.
static void ZcWorkerTick() {
    if (g_zcWorkerGen == g_zcGen) return;
    for (auto& s : g_zc) { ZcRel(s.tex11); s.opened = nullptr; }
    g_zcWorkerGen = g_zcGen;
}

// Bridge thread, inside ConsumeUpload with g_pubLock held SHARED.
static bool ZcWorkerTake(long idx) {
    if (idx < 0 || idx >= kZcSlots || g_zcWorkerGen != g_zcGen) return false;
    ZcSlot& s = g_zc[idx];
    if (InterlockedCompareExchange(&s.state, 3, 2) != 2) return false;
    HRESULT hr = S_OK;
    if (!g_zc11) {
        if (FAILED(hr = g_d3d11->QueryInterface(__uuidof(ID3D11Device5), (void**)&g_zc11)) ||
            FAILED(hr = g_ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&g_zcCtx)) ||
            FAILED(hr = g_zc11->OpenSharedFence(g_zcFH, __uuidof(ID3D11Fence), (void**)&g_zcF11)) ||
            FAILED(hr = g_zc11->OpenSharedFence(g_zcRH, __uuidof(ID3D11Fence), (void**)&g_zcR11))) {
            InterlockedExchange(&s.state, 0); ZcFail("bridge could not open the shared fences", hr); return false;
        }
    }
    if (!s.tex11 || s.opened != s.sharedH) {
        ZcRel(s.tex11);
        D3D11_TEXTURE2D_DESC td = {};
        hr = g_zc11->OpenSharedResource1(s.sharedH, __uuidof(ID3D11Texture2D), (void**)&s.tex11);
        if (SUCCEEDED(hr) && s.tex11) s.tex11->GetDesc(&td);
        if (FAILED(hr) || !s.tex11 || td.Width != g_w || td.Height != g_h) {
            ZcRel(s.tex11); InterlockedExchange(&s.state, 0);
            ZcFail(FAILED(hr) ? "bridge could not open a shared frame" : "shared frame has the wrong size", hr);
            return false;
        }
        s.opened = s.sharedH;
        static long s_logged = 0;
        if (InterlockedIncrement(&s_logged) == 1)
            XLOG("[OLVR][ZEROCOPY] bridge receiving frames on the GPU (%ux%u fmt=%d)", td.Width, td.Height, (int)td.Format);
    }
    g_zcCtx->Wait(g_zcF11, s.value);          // GPU-side: the game's copy into it has landed
    g_ctx->CopyResource(g_uploadTex, s.tex11);
    g_zcCtx->Signal(g_zcR11, s.value);        // and the game may reuse it after this copy
    g_ctx->Flush();
    return true;
}

// ---- [EXSHARE] (Test36) the hand-over for a D3D9Ex game device -----------------------------------
// With the game on D3D9Ex ([D3D9EX] in d3d9_proxy.cpp) a render target can be created SHARED and
// the bridge's D3D11 device opens the same memory. Per Present the backbuffer is StretchRect'ed
// into a free shared frame and a D3D9 event query is issued behind it; the next Present sees the
// query done and publishes that frame (the newest finished one). The bridge copies it out and
// frees the slot once its own D3D11 event says the copy has run. Neither thread ever waits on the
// GPU, and the game keeps native D3D9 speed: measured 81 fps against D3D9On12's 46 in the same
// spot. Slots: 0 free, 1 copy in flight, 2 published, 3 bridge copying out.
static const int kExSlots = 6;   // 4 ran dry at ~80 fps on the theater screen (copy-in + publish + copy-out)
struct ExSlot {
    IDirect3DTexture9* tex9 = nullptr;
    IDirect3DSurface9* surf9 = nullptr;
    IDirect3DQuery9*   done9 = nullptr;     // D3D9 event behind the copy into it
    HANDLE             sh = nullptr;        // legacy shared handle (owned by the D3D9 resource)
    volatile long      state = 0;
    uint64_t           order = 0;
    XrPosef            pose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
    long               poseValid = 0;
    float              fovX = 0.0f, fovY = 0.0f;
    long               lensFit = 0;
    ID3D11Texture2D*   tex11 = nullptr;     // bridge-owned
    HANDLE             opened = nullptr;    // bridge-owned
    ID3D11Query*       done11 = nullptr;    // bridge-owned: its copy out has run
};
static ExSlot        g_ex[kExSlots];
static volatile long g_exState = 0;          // 0 untried, 1 in use, -1 unavailable this session
static volatile long g_exGen = 0;
static long          g_exWorkerGen = -1;
static UINT          g_exW = 0, g_exH = 0;
static uint64_t      g_exOrder = 0;
static long          g_exSkipped = 0;
extern "C" int OLProxy_IsD3D9Ex();

static void ExFail(const char* why, long hr) {
    if (InterlockedExchange(&g_exState, -1) != -1)
        XLOG("[OLVR][EXSHARE] unavailable (%s, hr=0x%08lX) - frames go through the readback path", why, hr);
}

// Game (render) thread. Our D3D9 objects only; a D3D11 view the bridge opened keeps the memory alive
// until the bridge drops it on the generation change.
static void ExReleaseFrames() {
    AcquireSRWLockExclusive(&g_pubLock);
    InterlockedIncrement(&g_exGen);
    for (auto& s : g_ex) {
        if (s.done9) { s.done9->Release(); s.done9 = nullptr; }
        if (s.surf9) { s.surf9->Release(); s.surf9 = nullptr; }
        if (s.tex9)  { s.tex9->Release();  s.tex9 = nullptr; }
        s.sh = nullptr; InterlockedExchange(&s.state, 0);
    }
    g_pubExSlot = -1;
    g_exW = g_exH = 0;
    ReleaseSRWLockExclusive(&g_pubLock);
}

static bool ExEnsureFrames(IDirect3DDevice9* dev, UINT w, UINT h) {
    if (g_ex[0].surf9 && g_exW == w && g_exH == h) return true;
    ExReleaseFrames();
    HRESULT hr = S_OK;
    for (auto& s : g_ex) {
        HANDLE sh = nullptr;
        if (FAILED(hr = dev->CreateTexture(w, h, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &s.tex9, &sh)) ||
            !s.tex9 || !sh) { ExReleaseFrames(); ExFail("shared render target", hr); return false; }
        s.sh = sh;
        if (FAILED(hr = s.tex9->GetSurfaceLevel(0, &s.surf9)) ||
            FAILED(hr = dev->CreateQuery(D3DQUERYTYPE_EVENT, &s.done9))) {
            ExReleaseFrames(); ExFail("shared frame surface/query", hr); return false;
        }
    }
    g_exW = w; g_exH = h;
    XLOG("[OLVR][EXSHARE] %d shared frames %ux%u", kExSlots, w, h);
    return true;
}

// Game (render) thread, once per Present.
static bool PublishExShare(IDirect3DDevice9* dev, IDirect3DSurface9* bb) {
    const double t0 = PerfNowMs();
    // 1. Copies that have finished: publish the newest, older finished ones were never shown.
    ExSlot* done[kExSlots] = {}; int nd = 0;
    for (auto& s : g_ex)
        if (s.state == 1 && s.done9 && s.done9->GetData(nullptr, 0, 0) == S_OK) done[nd++] = &s;
    ExSlot* newest = nullptr;
    for (int i = 0; i < nd; ++i) if (!newest || done[i]->order > newest->order) newest = done[i];
    bool published = false;
    if (newest) {
        AcquireSRWLockExclusive(&g_pubLock);
        for (int i = 0; i < nd; ++i) if (done[i] != newest) InterlockedExchange(&done[i]->state, 0);
        // A published frame the bridge never took is superseded: free it.
        if (g_pubExSlot >= 0 && g_pubExSlot < kExSlots) InterlockedCompareExchange(&g_ex[g_pubExSlot].state, 0, 2);
        InterlockedExchange(&newest->state, 2);
        g_pubExSlot = (long)(newest - g_ex);
        g_pubZcSlot = -1;
        g_pubW = g_exW; g_pubH = g_exH; g_pubPitch = g_exW * 4;
        g_pubPose = newest->pose; g_pubPoseValid = newest->poseValid;
        g_pubFovX = newest->fovX; g_pubFovY = newest->fovY; g_pubLensFit = newest->lensFit;
        ++g_pubSeq;
        ReleaseSRWLockExclusive(&g_pubLock);
        published = true;
        if (g_exState == 0) InterlockedExchange(&g_exState, 1);
    }
    // 2. This frame's copy, tagged with the pose and FOV it was rendered with.
    ExSlot* slot = nullptr;
    for (auto& s : g_ex) if (s.state == 0) { slot = &s; break; }
    if (slot && SUCCEEDED(dev->StretchRect(bb, nullptr, slot->surf9, nullptr, D3DTEXF_NONE))) {
        slot->done9->Issue(D3DISSUE_END);
        AcquireSRWLockExclusive(&g_pubLock);
        TakeFrameTag(&slot->pose, &slot->poseValid, &slot->fovX, &slot->fovY, &slot->lensFit);
        ReleaseSRWLockExclusive(&g_pubLock);
        slot->order = ++g_exOrder;
        InterlockedExchange(&slot->state, 1);
    } else if (!slot) {
        ++g_exSkipped;
    }
    g_pfCopy.Add(PerfNowMs() - t0);
    AcquireSRWLockExclusive(&g_pubLock);
    PromoteAtPresent();
    ReleaseSRWLockExclusive(&g_pubLock);
    return published;
}

// Bridge thread: drop what was opened for an older generation, and free slots whose copy out ran.
static void ExWorkerTick() {
    if (g_exWorkerGen != g_exGen) {
        for (auto& s : g_ex) {
            if (s.tex11)  { s.tex11->Release();  s.tex11 = nullptr; }
            if (s.done11) { s.done11->Release(); s.done11 = nullptr; }
            s.opened = nullptr;
        }
        g_exWorkerGen = g_exGen;
        return;
    }
    for (auto& s : g_ex)
        if (s.state == 3 && s.done11 && g_ctx->GetData(s.done11, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK)
            InterlockedCompareExchange(&s.state, 0, 3);
}

// Bridge thread, inside ConsumeUpload with g_pubLock held SHARED.
static bool ExWorkerTake(long idx) {
    if (idx < 0 || idx >= kExSlots || g_exWorkerGen != g_exGen) return false;
    ExSlot& s = g_ex[idx];
    if (InterlockedCompareExchange(&s.state, 3, 2) != 2) return false;
    if (!s.tex11 || s.opened != s.sh) {
        if (s.tex11) { s.tex11->Release(); s.tex11 = nullptr; }
        D3D11_TEXTURE2D_DESC td = {};
        HRESULT hr = g_d3d11->OpenSharedResource(s.sh, __uuidof(ID3D11Texture2D), (void**)&s.tex11);
        if (SUCCEEDED(hr) && s.tex11) s.tex11->GetDesc(&td);
        if (FAILED(hr) || !s.tex11 || td.Width != g_w || td.Height != g_h) {
            if (s.tex11) { s.tex11->Release(); s.tex11 = nullptr; }
            InterlockedExchange(&s.state, 0);
            ExFail(FAILED(hr) ? "bridge could not open a shared frame" : "shared frame has the wrong size", hr);
            return false;
        }
        if (!s.done11) {
            D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
            g_d3d11->CreateQuery(&qd, &s.done11);
        }
        s.opened = s.sh;
        static long s_logged = 0;
        if (InterlockedIncrement(&s_logged) == 1)
            XLOG("[OLVR][EXSHARE] bridge receiving frames on the GPU (%ux%u fmt=%d)", td.Width, td.Height, (int)td.Format);
    }
    g_ctx->CopyResource(g_uploadTex, s.tex11);
    if (s.done11) g_ctx->End(s.done11); else InterlockedExchange(&s.state, 0);
    g_ctx->Flush();
    return true;
}

extern "C" void OLProxy_ExStatReport();   // (Test47) [EXSTAT] d3d9_proxy.cpp
static void LensPublish(IDirect3DDevice9* dev);   // [LENSCAM] below, with the flip screen
static bool PublishBackbuffer(IDirect3DDevice9* dev) {
    if (OLProxy_IsD3D9Ex()) LensPublish(dev);   // [LENSCAM] (Test46)
    IDirect3DSurface9* bb = OLProxy_AcquireNativeStereo();
    if (!bb && (FAILED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) || !bb)) return false;
    D3DSURFACE_DESC sd;
    if (FAILED(bb->GetDesc(&sd)) || sd.Width < 64 || sd.Height < 64) { bb->Release(); return false; }
    if (OLProxy_IsD3D9Ex() && g_exState >= 0 && ExEnsureFrames(dev, sd.Width, sd.Height) && g_exState >= 0) {
        const bool ok = PublishExShare(dev, bb);   // [EXSHARE]
        bb->Release();
        if (g_exState >= 0) return ok;
    }
    if (g_exW) ExReleaseFrames();
    if (!OLProxy_IsD3D9Ex() && g_zeroCopy && g_zcState >= 0 && ZcInit(dev) && ZcEnsureFrames(dev, sd.Width, sd.Height) && g_zcState >= 0) {
        const bool ok = PublishZeroCopy(dev, bb);
        bb->Release();
        if (g_zcState >= 0) return ok;
    }
    if (g_zcW) ZcReleaseFrames();   // failed over: nothing of it may stay published
    if (g_asyncReadback) {
        const bool ok = PublishBackbufferAsync(dev, bb, sd);
        bb->Release();
        if (ok || g_rs[0].surf) return ok;
        g_asyncReadback = false;   // the ring could not be created: same-frame readback from now on
    }
    static UINT s_w = 0, s_h = 0; static D3DFORMAT s_fmt = D3DFMT_UNKNOWN;
    if (!g_sysSurf || s_w != sd.Width || s_h != sd.Height || s_fmt != sd.Format) {
        if (g_sysSurf) { g_sysSurf->Release(); g_sysSurf = nullptr; }
        if (SUCCEEDED(dev->CreateOffscreenPlainSurface(sd.Width, sd.Height, sd.Format, D3DPOOL_SYSTEMMEM, &g_sysSurf, nullptr))) {
            s_w = sd.Width; s_h = sd.Height; s_fmt = sd.Format;
            XLOG("[OLVR][XR] readback surface %ux%u fmt=%d", sd.Width, sd.Height, (int)sd.Format);
        } else { g_sysSurf = nullptr; bb->Release(); return false; }
    }
    bool ok = false;
    const double t0 = PerfNowMs();
    if (SUCCEEDED(dev->GetRenderTargetData(bb, g_sysSurf))) {
        const double t1 = PerfNowMs();
        g_pfGrab.Add(t1 - t0);
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(g_sysSurf->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
            AcquireSRWLockExclusive(&g_pubLock);
            CopyToPublished(lr, sd.Width, sd.Height);
            g_pubZcSlot = -1; g_pubExSlot = -1;
            // [POSEFREEZE] Tag the finished image with the pose it was ACTUALLY rendered with -
            // the frozen live value both eyes just used - then promote the pending pose.
            TakeFrameTag(&g_pubPose, &g_pubPoseValid, &g_pubFovX, &g_pubFovY, &g_pubLensFit);   // [FRAMETAG]
            ++g_pubSeq;
            PromoteAtPresent();
            ReleaseSRWLockExclusive(&g_pubLock);
            g_sysSurf->UnlockRect();
            g_pfCopy.Add(PerfNowMs() - t1);
            ok = true;
        }
    }
    bb->Release();
    return ok;
}

// [SUBMITTHREAD] Worker-side half: adopt the newest published frame. Swapchain rebuilds happen
// HERE, from the published size - the worker owns every D3D11 and OpenXR object.
// Bridge thread: save g_uploadTex as a PNG (24-bit BGR) for "xrsnap".
static void DevSaveUploadTex() {
    if (!g_devXrSnapReq || !g_uploadTex || !g_conSeq) return;
    InterlockedExchange(&g_devXrSnapReq, 0);
    D3D11_TEXTURE2D_DESC td; g_uploadTex->GetDesc(&td);
    td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    ID3D11Texture2D* st = nullptr;
    bool ok = false;
    if (SUCCEEDED(g_d3d11->CreateTexture2D(&td, nullptr, &st)) && st) {
        g_ctx->CopyResource(st, g_uploadTex);
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
            IWICImagingFactory* f = nullptr;
            const HRESULT ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f))) && f) {
                IWICStream* s = nullptr; IWICBitmapEncoder* e = nullptr; IWICBitmapFrameEncode* fr = nullptr;
                if (SUCCEEDED(f->CreateStream(&s)) && SUCCEEDED(s->InitializeFromFilename(g_devXrSnapPath, GENERIC_WRITE)) &&
                    SUCCEEDED(f->CreateEncoder(GUID_ContainerFormatPng, nullptr, &e)) && SUCCEEDED(e->Initialize(s, WICBitmapEncoderNoCache)) &&
                    SUCCEEDED(e->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize(td.Width, td.Height))) {
                    WICPixelFormatGUID pf = GUID_WICPixelFormat24bppBGR;
                    fr->SetPixelFormat(&pf);
                    std::vector<BYTE> px((size_t)td.Width * 3 * td.Height);
                    for (UINT y = 0; y < td.Height; ++y) {
                        const BYTE* sp = (const BYTE*)m.pData + (size_t)y * m.RowPitch;
                        BYTE* dp = px.data() + (size_t)y * td.Width * 3;
                        for (UINT xx = 0; xx < td.Width; ++xx, sp += 4, dp += 3) { dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2]; }
                    }
                    ok = SUCCEEDED(fr->WritePixels(td.Height, td.Width * 3, (UINT)px.size(), px.data())) &&
                         SUCCEEDED(fr->Commit()) && SUCCEEDED(e->Commit());
                }
                if (fr) fr->Release(); if (e) e->Release(); if (s) s->Release();
                f->Release();
            }
            if (SUCCEEDED(ci)) CoUninitialize();
            g_ctx->Unmap(st, 0);
        }
        st->Release();
    }
    XLOG("[OLVR][DEVBRIDGE] xrsnap %ls %ux%u -> %s", g_devXrSnapPath, td.Width, td.Height, ok ? "saved" : "FAILED");
}

static void ConsumeUpload() {
    DevSaveUploadTex();
    ZcWorkerTick();
    ExWorkerTick();   // [EXSHARE]
    AcquireSRWLockShared(&g_pubLock);
    const uint64_t seq = g_pubSeq; const UINT w = g_pubW, h = g_pubH;
    ReleaseSRWLockShared(&g_pubLock);
    if (!seq || seq == g_conSeq) return;
    if (w != g_w || h != g_h) { if (!RebuildForSize(w, h)) return; }
    AcquireSRWLockShared(&g_pubLock);
    if (g_pubZcSlot >= 0) {
        // [ZEROCOPY] the frame is on the GPU already
        if (g_pubW == g_w && g_pubH == g_h && g_uploadTex && g_pubSeq == seq) {
            const double t0 = PerfNowMs();
            if (ZcWorkerTake(g_pubZcSlot)) {
                g_conSeq = g_pubSeq; g_conPose = g_pubPose; g_conPoseValid = g_pubPoseValid;
                g_conFovX = g_pubFovX; g_conFovY = g_pubFovY; g_conLensFit = g_pubLensFit;
                const double dt = PerfNowMs() - t0;
                AcquireSRWLockExclusive(&g_pfLock); g_pfUpload.Add(dt); ++g_pfXrNew; ReleaseSRWLockExclusive(&g_pfLock);
            }
        }
    } else if (g_pubExSlot >= 0) {
        // [EXSHARE] the frame is in shared GPU memory
        if (g_pubW == g_w && g_pubH == g_h && g_uploadTex && g_pubSeq == seq) {
            const double t0 = PerfNowMs();
            if (ExWorkerTake(g_pubExSlot)) {
                g_conSeq = g_pubSeq; g_conPose = g_pubPose; g_conPoseValid = g_pubPoseValid;
                g_conFovX = g_pubFovX; g_conFovY = g_pubFovY; g_conLensFit = g_pubLensFit;
                const double dt = PerfNowMs() - t0;
                AcquireSRWLockExclusive(&g_pfLock); g_pfUpload.Add(dt); ++g_pfXrNew; ReleaseSRWLockExclusive(&g_pfLock);
            }
        }
    } else if (g_pubW == g_w && g_pubH == g_h && g_uploadTex && !g_pubPixels.empty()) {
        const double t0 = PerfNowMs();
        if (g_devUpload) g_ctx->UpdateSubresource(g_uploadTex, 0, nullptr, g_pubPixels.data(), g_pubPitch, 0);
        g_conSeq = g_pubSeq; g_conPose = g_pubPose; g_conPoseValid = g_pubPoseValid;
        g_conFovX = g_pubFovX; g_conFovY = g_pubFovY; g_conLensFit = g_pubLensFit;
        const double dt = PerfNowMs() - t0;
        AcquireSRWLockExclusive(&g_pfLock); g_pfUpload.Add(dt); ++g_pfXrNew; ReleaseSRWLockExclusive(&g_pfLock);
    }
    ReleaseSRWLockShared(&g_pubLock);
}

extern "C" int  OLUE3_StereoMode();          // 0 none, 1 SBS split, 2 SFR, 3 AER
extern "C" int  OLUE3_HeadRollEnabled();     // one flag for camera AND declared pose (see below)
extern "C" int  OLUE3_GetEyeSwap();          // the Mass Effect model flips the routing with it too
extern "C" int  OLUE3_GetHeadPos();          // [HEADPOS] lean/duck switch, for the periodic log
extern "C" int  OLUE3_CurrentEye();          // AER: which eye this frame rendered (0=L, 1=R)
extern "C" IDirect3DSurface9* OLProxy_LeftEye();
extern "C" int  OLUE3_SplitActive();
extern "C" void OLUE3_FrameTick();
extern "C" void OLUE3_GetEyeFov(float* fx, float* fy);

// One swapchain per eye. SBS packs both eyes into one frame, so each eye is half-width; SFR and
// AER render a full frame per eye, so each eye is the full backbuffer.
//
// The eye width therefore depends on the MODE, and the mode is switchable at runtime (NUM2/NUM4).
// Latching it once was a real bug: after a split->AER switch the swapchains kept the half width,
// and CopyResource between mismatched sizes silently does NOTHING - one eye freezes or goes black
// with no error anywhere. Rebuild whenever the mode changes.
static int g_eyeMode = -1;
static int g_eyeSlack = 0;          // convergence headroom currently baked into g_eyeW

// Slack the current convergence needs, in pixels, quantized to 32 so dragging the slider does
// not rebuild the swapchains at every notch. Capped so the eye always keeps 2/3 of its width.
static int WantConvSlackPx() {
    // [STEREOMODEL] the Mass Effect model reserves nothing: it samples with a clamping sampler,
    // so the eye keeps the full half at every convergence setting.
    if (g_meModel) return 0;
    const int halfW = (int)(g_w / 2);
    int need = (int)ceilf(fabsf(g_convShift) * kConvUnit * (float)halfW);
    if (need <= 0) return 0;
    need = ((need + 31) / 32) * 32;
    const int cap = halfW / 3;
    return need > cap ? cap : need;
}

static bool EnsureEyeSwapchains() {
    const int m = OLUE3_StereoMode();
    const int wantSlack = (m == 1) ? WantConvSlackPx() : 0;
    if (g_scL && g_scR) {
        if (m == g_eyeMode && wantSlack == g_eyeSlack) return true;
        XLOG("[OLVR][XR] stereo mode %d -> %d, rebuilding per-eye swapchains", g_eyeMode, m);
        DestroyEyeSwapchains();
        for (int e = 0; e < 2; ++e) { if (g_eyeHold[e]) { g_eyeHold[e]->Release(); g_eyeHold[e] = nullptr; } }
    }
    // The slack must be part of this guard too: a convergence-only change leaves the mode equal,
    // and without it the rebuild would be skipped and the eye width would silently disagree with
    // the copy box - which is the "one eye freezes" failure described above.
    if (g_eyeInitTried && m == g_eyeMode && wantSlack == g_eyeSlack) return false;
    g_eyeInitTried = true;
    g_eyeMode = m;
    // CONVERGENCE HEADROOM, only when it is being used. Each rendered half is exactly one eye
    // wide, so a convergence shift has nowhere to go: sliding the window pulls the OTHER eye's
    // pixels in at the edge, which is what made convergence feel broken. ME1 gets away with a
    // plain shift because it samples with a clamping sampler; this mod does box copies instead,
    // so the room is reserved ahead of time. At convergence 0 the slack is 0 and the eye gets the FULL half - no sharpness
    // is paid for a feature that is switched off. Crossing 0 rebuilds the swapchains once.
    g_eyeSlack = wantSlack;
    g_srcEyeW = (m == 1) ? (g_w / 2 - 2 * g_eyeSlack) : g_w;
    g_srcEyeH = g_h;
    g_eyeW = (g_nativeEyeResolution && g_runtimeEyeW) ? g_runtimeEyeW : g_srcEyeW;
    g_eyeH = (g_nativeEyeResolution && g_runtimeEyeH) ? g_runtimeEyeH : g_srcEyeH;
    XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    ci.sampleCount = 1; ci.width = g_eyeW; ci.height = g_eyeH;
    ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
    XrSwapchain* scs[2] = { &g_scL, &g_scR };
    std::vector<ID3D11Texture2D*>* vecs[2] = { &g_imgL, &g_imgR };
    for (int e = 0; e < 2; ++e) {
        if (!XrSucceeded(g_fn.createSwapchain(g_session, &ci, scs[e]))) {
            XLOG("[OLVR][XR] eye swapchain %d create FAILED %ux%u", e, g_eyeW, g_eyeH);
            g_scL = g_scR = 0; return false;
        }
        uint32_t n = 0; g_fn.enumerateSwapchainImages(*scs[e], 0, &n, nullptr);
        std::vector<XrSwapchainImageD3D11KHR> imgs(n);
        for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
        if (!XrSucceeded(g_fn.enumerateSwapchainImages(*scs[e], n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) {
            g_scL = g_scR = 0; return false;
        }
        for (auto& im : imgs) vecs[e]->push_back(im.texture);
        // [GAMMAFMT] The format the runtime ACTUALLY backs the swapchain with decides whether
        // the compositor degammas these bytes on sampling. VDXR was measured substituting the
        // typeless family for the requested sRGB format and still looking right; if SteamVR resolves the
        // same bytes through a different view, that is one uniform gamma step - the "a little
        // dark" report. This line is the ground truth for that comparison.
        if (e == 0 && !imgs.empty() && imgs[0].texture) {
            D3D11_TEXTURE2D_DESC dd; imgs[0].texture->GetDesc(&dd);
            XLOG("[OLVR][GAMMAFMT] eye swapchain: asked fmt=%d B8G8R8A8_UNORM_SRGB, runtime allocated fmt=%d%s",
                 (int)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, (int)dd.Format,
                 dd.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ? " (exact match)" : " (substituted)");
        }
    }
    XLOG("[OLVR][XR] per-eye swapchains ready %ux%u from %ux%u source (SBS split detected)",
         g_eyeW, g_eyeH, g_srcEyeW, g_srcEyeH);
    return true;
}

// ---- [GAMMA] corrected blit ---------------------------------------------------------------
// "The colors are weird, the blacks are not good" (2026-07-23) has two real halves:
//  1. The game's brightness calibration goes through SetGammaRamp - a MONITOR output curve.
//     This capture reads the backbuffer BEFORE it, so the headset never saw that calibration.
//  2. The bytes are tagged sRGB for the compositor, but the game was authored against ~2.2
//     displays. The two curves differ exactly in the toe - the BLACKS.
// Fix = ME1's shape: a Gamma knob (ME1 m8_menu.cpp: 0.5..2.5, default 1.0) applied at the
// blit, with the captured game ramp folded in. Both live in one 256-entry LUT. The LUT stores
// LINEAR values (the sRGB render target re-encodes on write, so storing gamma bytes would
// double-encode - washed out), which also means identity-LUT == plain copy, bit for bit.
// When gamma == 1.0 and no ramp is set, the shader path is bypassed entirely and the blits
// stay the exact CopyResource calls of the confirmed baseline.
extern "C" int OLProxy_GammaRampGen();
extern "C" int OLProxy_GetGammaRamp(unsigned short* r, unsigned short* g, unsigned short* b);
extern "C" int OLProxy_NightVisionActive();

// RANGE, and why it is not the obvious one (2026-08-03, after a live blinding). The old slider
// ran gamma 0.5 to 2.5, and gamma ABOVE 1 darkens. Outlast is already a nearly black game, so a
// small nudge upward crushed the whole picture, including this menu, to unreadable - and there
// was no way back that did not involve reading a slider you could no longer see.
// Two fixes: the dark half is capped where it is still a toe adjustment rather than a blackout,
// and the menu now presents this as BRIGHTNESS (1/gamma), so the slider moves the same way the
// picture does. An out-of-range value from an older ini clamps in here rather than being obeyed.
static const float kGammaMin = 0.50f;   // brightest
static const float kGammaMax = 1.20f;   // darkest that can still be seen through
static volatile float g_gamma = 1.0f;
extern "C" float OLXR_GetGamma() { return g_gamma; }
extern "C" void  OLXR_SetGamma(float v) {
    if (!(v > 0.0f) || !isfinite(v)) v = 1.0f;
    if (v < kGammaMin) v = kGammaMin; if (v > kGammaMax) v = kGammaMax;
    g_gamma = v;
}
extern "C" void  OLXR_ResetGamma() { g_gamma = 1.0f; }
extern "C" int OLXR_GammaRampApplied() { return OLProxy_GammaRampGen() ? 1 : 0; }

static ID3D11VertexShader*       g_blitVS = nullptr;
static ID3D11PixelShader*        g_blitPS = nullptr;
static ID3D11Buffer*             g_blitCB = nullptr;
static ID3D11SamplerState*       g_blitSamp = nullptr;
static ID3D11Texture2D*          g_lutTex = nullptr;
static ID3D11ShaderResourceView* g_lutSRV = nullptr;
static int   g_lutGen = -1;
static float g_lutGamma = -999.0f;
static int   g_lutNightVision = -1;
static volatile float g_nightVisionBrightness = 0.72f;
static bool  g_blitInitTried = false, g_blitReady = false;
static ID3D11BlendState*         g_blitBlend = nullptr;
static ID3D11DepthStencilState*  g_blitDepth = nullptr;
static ID3D11RasterizerState*    g_blitRaster = nullptr;
// The blit's own render target. This NEVER draws straight into an OpenXR swapchain image: every
// confirmed-good build here and in ME1/ME2 only ever COPIES into those, and the direct-render
// version scrambled the headset picture while every D3D call reported success.
static ID3D11Texture2D*          g_blitTmp = nullptr;
static ID3D11RenderTargetView*   g_blitTmpRtv = nullptr;
static UINT g_blitTmpW = 0, g_blitTmpH = 0;

static bool GammaActive() {
    return fabsf(g_gamma - 1.0f) > 0.005f || OLProxy_GammaRampGen() != 0 ||
           OLProxy_NightVisionActive() != 0;
}

typedef HRESULT (WINAPI *PFN_D3DCompile)(LPCVOID, SIZE_T, LPCSTR, const void*, void*,
                                         LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

// clampU.xy bounds the sampled u inside this eye's OWN half of the backbuffer. The Mass Effect
// model slides the sampled window past the half's edge on purpose; clamping repeats the edge
// column instead of letting the left eye pull in the right eye's pixels at the seam. That clamp
// is exactly what lets the shift cost no resolution and reserve no slack.
// [HIRES] Test35: tune.w = 0 when the eye is drawn 1:1 (the game now renders at the headset's own
// size), and the bicubic's 13 taps collapse to one - at 1:1 it would only reproduce the texel.
// [UPSCALE] Test34: tune.y selects the upscaler. 0 is the Test28 path exactly (bilinear plus a
// 4-tap unsharp mask). 1 is a Catmull-Rom bicubic in 9 bilinear taps, clamped to the 2x2 source
// texels around the point so its negative lobes cannot ring, followed by contrast-adaptive
// sharpening (the CAS idea: sharpen hard where local contrast is low, back off near edges that
// would halo). Every tap goes through the same seam clamp as before.
static const char* kBlitHlsl =
"cbuffer CB : register(b0) { float4 box; float4 clampU; float4 tune; };\n" // u0,v0,du,dv | minU,maxU,texel | sharp,mode
"struct VOut { float4 p : SV_Position; float2 uv : TEXCOORD0; };\n"
"VOut vs(uint id : SV_VertexID) {\n"
"  float2 t = float2((id << 1) & 2, id & 2);\n"                      // fullscreen triangle
"  VOut o; o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"  o.uv = box.xy + t * box.zw; return o; }\n"
"Texture2D srcT : register(t0);\n"
"Texture2D lutT : register(t1);\n"
"SamplerState s0 : register(s0);\n"
"float3 Tap(float2 uv) {\n"
"  uv = float2(clamp(uv.x, clampU.x, clampU.y), clamp(uv.y, clampU.w * 0.5, 1.0 - clampU.w * 0.5));\n"
"  return srcT.SampleLevel(s0, uv, 0).rgb; }\n"
"float3 CatmullRom(float2 uv) {\n"
"  float2 tx = clampU.zw;\n"
"  float2 sp = uv / tx;\n"
"  float2 t1 = floor(sp - 0.5) + 0.5;\n"
"  float2 f = sp - t1;\n"
"  float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));\n"
"  float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);\n"
"  float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));\n"
"  float2 w3 = f * f * (-0.5 + 0.5 * f);\n"
"  float2 w12 = w1 + w2;\n"
"  float2 p0 = (t1 - 1.0) * tx, p3 = (t1 + 2.0) * tx, p12 = (t1 + w2 / w12) * tx;\n"
"  float3 r = (Tap(float2(p0.x, p0.y)) * w0.x + Tap(float2(p12.x, p0.y)) * w12.x + Tap(float2(p3.x, p0.y)) * w3.x) * w0.y\n"
"           + (Tap(float2(p0.x, p12.y)) * w0.x + Tap(float2(p12.x, p12.y)) * w12.x + Tap(float2(p3.x, p12.y)) * w3.x) * w12.y\n"
"           + (Tap(float2(p0.x, p3.y)) * w0.x + Tap(float2(p12.x, p3.y)) * w12.x + Tap(float2(p3.x, p3.y)) * w3.x) * w3.y;\n"
"  float2 a = t1 * tx, b = (t1 + 1.0) * tx;\n"
"  float3 q0 = Tap(a), q1 = Tap(float2(b.x, a.y)), q2 = Tap(float2(a.x, b.y)), q3 = Tap(b);\n"
"  return clamp(r, min(min(q0, q1), min(q2, q3)), max(max(q0, q1), max(q2, q3))); }\n"
"float3 Cas(float3 c, float2 uv, float sharp) {\n"
"  float2 tx = clampU.zw;\n"
"  float3 n = Tap(uv + float2(0, -tx.y)), s = Tap(uv + float2(0, tx.y));\n"
"  float3 e = Tap(uv + float2(tx.x, 0)), w = Tap(uv + float2(-tx.x, 0));\n"
"  float3 mn = min(c, min(min(n, s), min(e, w)));\n"
"  float3 mx = max(c, max(max(n, s), max(e, w)));\n"
"  float3 amp = sqrt(saturate(min(mn, 1.0 - mx) / max(mx, 0.0001)));\n"
"  float3 k = -amp * (0.2 * sharp);\n"                               // 0 = none .. 1 = CAS's strongest
"  return saturate((c + (n + s + e + w) * k) / (1.0 + 4.0 * k)); }\n"
"float4 ps(VOut i) : SV_Target {\n"
"  float2 uv = float2(clamp(i.uv.x, clampU.x, clampU.y), i.uv.y);\n"
"  float3 c;\n"
"  if (tune.y > 0.5) {\n"
"    c = tune.w > 0.5 ? CatmullRom(uv) : Tap(uv);\n"                  // 1:1 needs no resampling
"    if (tune.x > 0.0001) c = Cas(c, uv, tune.x);\n"
"  } else {\n"
"  c = srcT.Sample(s0, uv).rgb;\n"                                   // gamma-space bytes
"  if (tune.x > 0.0001) {\n"
"    float2 tx = clampU.zw;\n"
"    float2 ul = float2(clamp(uv.x-tx.x,clampU.x,clampU.y),clamp(uv.y,tx.y*0.5,1.0-tx.y*0.5));\n"
"    float2 ur = float2(clamp(uv.x+tx.x,clampU.x,clampU.y),clamp(uv.y,tx.y*0.5,1.0-tx.y*0.5));\n"
"    float2 uu = float2(uv.x,clamp(uv.y-tx.y,tx.y*0.5,1.0-tx.y*0.5));\n"
"    float2 ud = float2(uv.x,clamp(uv.y+tx.y,tx.y*0.5,1.0-tx.y*0.5));\n"
"    float3 blur = (srcT.Sample(s0,ul).rgb + srcT.Sample(s0,ur).rgb +\n"
"                   srcT.Sample(s0,uu).rgb + srcT.Sample(s0,ud).rgb) * 0.25;\n"
"    c = saturate(c + (c - blur) * tune.x);\n"
"  }\n"
"  }\n"
"  float3 o;\n"
"  o.r = lutT.Sample(s0, float2((c.r * 255.0 + 0.5) / 256.0, 0.5)).r;\n"
"  o.g = lutT.Sample(s0, float2((c.g * 255.0 + 0.5) / 256.0, 0.5)).g;\n"
"  o.b = lutT.Sample(s0, float2((c.b * 255.0 + 0.5) / 256.0, 0.5)).b;\n"
"  return float4(o * (1.0 - tune.z), 1.0); }\n";

static bool EnsureGammaPipeline() {
    if (g_blitReady) return true;
    if (g_blitInitTried) return false;
    g_blitInitTried = true;
    HMODULE dc = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3DCompile compile = dc ? (PFN_D3DCompile)GetProcAddress(dc, "D3DCompile") : nullptr;
    if (!compile) { XLOG("[OLVR][GAMMA] d3dcompiler_47 unavailable - gamma disabled, plain copy"); return false; }
    ID3DBlob* vsb = nullptr; ID3DBlob* psb = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(compile(kBlitHlsl, strlen(kBlitHlsl), nullptr, nullptr, nullptr, "vs", "vs_4_0", 0, 0, &vsb, &err)) ||
        FAILED(compile(kBlitHlsl, strlen(kBlitHlsl), nullptr, nullptr, nullptr, "ps", "ps_4_0", 0, 0, &psb, &err))) {
        XLOG("[OLVR][GAMMA] shader compile FAILED%s%s", err ? ": " : "", err ? (const char*)err->GetBufferPointer() : "");
        if (vsb) vsb->Release(); if (psb) psb->Release(); if (err) err->Release();
        return false;
    }
    bool ok = SUCCEEDED(g_d3d11->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_blitVS)) &&
              SUCCEEDED(g_d3d11->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_blitPS));
    vsb->Release(); psb->Release();
    if (ok) {
        D3D11_BUFFER_DESC bd = {}; bd.ByteWidth = 48; bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        ok = SUCCEEDED(g_d3d11->CreateBuffer(&bd, nullptr, &g_blitCB));
    }
    if (ok) {
        D3D11_SAMPLER_DESC sdc = {}; sdc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sdc.AddressU = sdc.AddressV = sdc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        ok = SUCCEEDED(g_d3d11->CreateSamplerState(&sdc, &g_blitSamp));
    }
    // Explicit pipeline state for the blit draw. The first version trusted context defaults and
    // the image came out scrambled; ME1's blit sets everything it depends on, so this does too.
    if (ok) {
        D3D11_BLEND_DESC bld = {};
        bld.RenderTarget[0].BlendEnable = FALSE;
        bld.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        ok = SUCCEEDED(g_d3d11->CreateBlendState(&bld, &g_blitBlend));
    }
    if (ok) {
        D3D11_DEPTH_STENCIL_DESC dsd = {};                 // depth and stencil both off
        ok = SUCCEEDED(g_d3d11->CreateDepthStencilState(&dsd, &g_blitDepth));
    }
    if (ok) {
        D3D11_RASTERIZER_DESC rsd = {};
        rsd.FillMode = D3D11_FILL_SOLID; rsd.CullMode = D3D11_CULL_NONE; rsd.DepthClipEnable = TRUE;
        ok = SUCCEEDED(g_d3d11->CreateRasterizerState(&rsd, &g_blitRaster));
    }
    g_blitReady = ok;
    XLOG("[OLVR][GAMMA] corrected-blit pipeline %s", ok ? "ready" : "FAILED - plain copy");
    return ok;
}

static float SrgbToLinear(float s) {
    return (s <= 0.04045f) ? s / 12.92f : powf((s + 0.055f) / 1.055f, 2.4f);
}

// LUT[i] = linear( pow(ramp[i], gamma) ). Rebuilt only when the slider or the game ramp moves.
static bool BuildGammaLut() {
    unsigned short rr[256], rg[256], rb[256];
    const int   gen = OLProxy_GetGammaRamp(rr, rg, rb);
    const float gam = g_gamma;
    const int nv = OLProxy_NightVisionActive() ? 1 : 0;
    if (g_lutSRV && gen == g_lutGen && nv == g_lutNightVision &&
        fabsf(gam - g_lutGamma) < 0.001f) return true;
    const float nvScale = nv ? g_nightVisionBrightness : 1.0f;
    float data[256 * 4];
    for (int i = 0; i < 256; ++i) {
        const float br = gen ? rr[i] / 65535.0f : i / 255.0f;
        const float bg = gen ? rg[i] / 65535.0f : i / 255.0f;
        const float bb = gen ? rb[i] / 65535.0f : i / 255.0f;
        data[i * 4 + 0] = SrgbToLinear(powf(br, gam)) * nvScale;
        data[i * 4 + 1] = SrgbToLinear(powf(bg, gam)) * nvScale;
        data[i * 4 + 2] = SrgbToLinear(powf(bb, gam)) * nvScale;
        data[i * 4 + 3] = 1.0f;
    }
    if (g_lutSRV) { g_lutSRV->Release(); g_lutSRV = nullptr; }
    if (g_lutTex) { g_lutTex->Release(); g_lutTex = nullptr; }
    D3D11_TEXTURE2D_DESC td = {}; td.Width = 256; td.Height = 1; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = { data, 256 * 16, 0 };
    if (FAILED(g_d3d11->CreateTexture2D(&td, &sd, &g_lutTex)) ||
        FAILED(g_d3d11->CreateShaderResourceView(g_lutTex, nullptr, &g_lutSRV))) {
        XLOG("[OLVR][GAMMA] LUT create FAILED"); return false;
    }
    g_lutGen = gen; g_lutGamma = gam; g_lutNightVision = nv;
    XLOG("[OLVR][GAMMA] LUT rebuilt: gamma=%.2f ramp=%s nightVision=%s scale=%.2f",
         gam, gen ? "game" : "none", nv ? "on" : "off", nvScale);
    return true;
}

// The blit draw's private render target, recreated only when the requested size changes.
static bool EnsureBlitTmp(UINT w, UINT h) {
    if (g_blitTmp && g_blitTmpRtv && g_blitTmpW == w && g_blitTmpH == h) return true;
    if (g_blitTmpRtv) { g_blitTmpRtv->Release(); g_blitTmpRtv = nullptr; }
    if (g_blitTmp)    { g_blitTmp->Release();    g_blitTmp = nullptr; }
    D3D11_TEXTURE2D_DESC td = {}; td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(g_d3d11->CreateTexture2D(&td, nullptr, &g_blitTmp))) { g_blitTmp = nullptr; return false; }
    if (FAILED(g_d3d11->CreateRenderTargetView(g_blitTmp, nullptr, &g_blitTmpRtv))) {
        g_blitTmp->Release(); g_blitTmp = nullptr; g_blitTmpRtv = nullptr; return false;
    }
    g_blitTmpW = w; g_blitTmpH = h;
    return true;
}

// Draw src[box] -> a private target with the LUT applied, then COPY that into dst (the acquired
// swapchain image). ME1's shape exactly. boxX is SIGNED and fractional so the Mass Effect model
// can slide the window past the half's edge; clampLoPx/clampHiPx bound the sample to this eye's
// own half. Views on src are made per call - caching by texture pointer is a use-after-free trap
// when swapchains rebuild.
// dstVp* place the drawn image inside dst, and boxY offsets the source vertically. Both exist
// for [FITFRUSTUM]: when the rendered picture covers a NARROWER angle than the eye the runtime
// is compositing, it must sit at its true position inside that eye with black around it, not
// be stretched to fill. A zero-offset, full-size call is byte-identical to the old behaviour.
static bool GammaDraw(ID3D11Texture2D* src, UINT srcW, UINT srcH,
                      float boxX, UINT boxW, UINT boxH,
                      float clampLoPx, float clampHiPx,
                      ID3D11Texture2D* dst, UINT dstW, UINT dstH,
                      float boxY = 0.0f,
                      float dstVpX = 0.0f, float dstVpY = 0.0f,
                      float dstVpW = 0.0f, float dstVpH = 0.0f) {
    if (!EnsureBlitTmp(dstW, dstH)) return false;
    ID3D11ShaderResourceView* srv = nullptr;
    if (FAILED(g_d3d11->CreateShaderResourceView(src, nullptr, &srv))) return false;
    // Half-texel inset so linear filtering can never blend across the SBS seam.
    const bool scalingUp = dstW > boxW || dstH > boxH;
    const float cb[12] = { boxX / (float)srcW, boxY / (float)srcH,
                           (float)boxW / (float)srcW, (float)boxH / (float)srcH,
                           (clampLoPx + 0.5f) / (float)srcW, (clampHiPx - 0.5f) / (float)srcW,
                           1.0f / (float)srcW, 1.0f / (float)srcH,
                           scalingUp ? g_upscaleSharpness : 0.0f, g_upscaler ? 1.0f : 0.0f,
                           1.0f - g_presentFade, scalingUp ? 1.0f : 0.0f };   // [FADE] tune.z = how dark, tune.w = resample
    g_ctx->UpdateSubresource(g_blitCB, 0, nullptr, cb, 0, 0);
    const bool placed = (dstVpW > 0.5f && dstVpH > 0.5f);
    D3D11_VIEWPORT vp = placed ? D3D11_VIEWPORT{ dstVpX, dstVpY, dstVpW, dstVpH, 0.0f, 1.0f }
                               : D3D11_VIEWPORT{ 0.0f, 0.0f, (float)dstW, (float)dstH, 0.0f, 1.0f };
    if (placed) {
        // Everything outside the placed picture must be opaque black, not last frame's pixels.
        const float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        g_ctx->ClearRenderTargetView(g_blitTmpRtv, black);
    }
    const float blendFactor[4] = { 0, 0, 0, 0 };
    ID3D11ShaderResourceView* srvs[2] = { srv, g_lutSRV };
    g_ctx->IASetInputLayout(nullptr);
    g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_ctx->VSSetShader(g_blitVS, nullptr, 0);
    g_ctx->VSSetConstantBuffers(0, 1, &g_blitCB);
    g_ctx->PSSetShader(g_blitPS, nullptr, 0);
    // The PS reads clampU from this same buffer, so it MUST be bound to the pixel stage as well.
    // Missing this line was the entire "scrambled horizontally" bug: an unbound PS constant buffer
    // reads zero, so clamp(u,0,0) pinned every sample to column 0.
    g_ctx->PSSetConstantBuffers(0, 1, &g_blitCB);
    g_ctx->PSSetShaderResources(0, 2, srvs);
    g_ctx->PSSetSamplers(0, 1, &g_blitSamp);
    g_ctx->OMSetBlendState(g_blitBlend, blendFactor, 0xffffffff);
    g_ctx->OMSetDepthStencilState(g_blitDepth, 0);
    g_ctx->OMSetRenderTargets(1, &g_blitTmpRtv, nullptr);
    g_ctx->RSSetState(g_blitRaster);
    g_ctx->RSSetViewports(1, &vp);
    g_ctx->Draw(3, 0);
    ID3D11ShaderResourceView* nul[2] = { nullptr, nullptr };
    ID3D11RenderTargetView* nulrt = nullptr;
    g_ctx->PSSetShaderResources(0, 2, nul);
    g_ctx->OMSetRenderTargets(1, &nulrt, nullptr);
    srv->Release();
    g_ctx->CopyResource(dst, g_blitTmp);
    return true;
}

// True when the shader path can run at all.
static bool ShaderBlitReady() {
    return EnsureGammaPipeline() && BuildGammaLut();
}
static bool GammaBlitReady() {
    return GammaActive() && ShaderBlitReady();
}

// Copy one eye's window of the captured backbuffer into that eye's swapchain image.
// srcX is SIGNED and fractional so the Mass Effect model can slide it past the half's edge;
// clampLo/clampHi bound the sample inside that eye's own half. In the Outlast model the window
// is a plain in-bounds box and this is the baseline's exact CopySubresourceRegion.
static bool BlitHalfToEye(XrSwapchain sc, std::vector<ID3D11Texture2D*>& imgs,
                          float srcX, float clampLoPx, float clampHiPx) {
    if (!sc || imgs.empty()) return false;
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (!XrSucceeded(g_fn.acquireSwapchainImage(sc, &ai, &idx))) return false;
    bool ok = false;
    XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
    if (XrSucceeded(g_fn.waitSwapchainImage(sc, &wi))) {
        const bool shifted = fabsf(srcX - clampLoPx) > 0.01f;
        // ALWAYS the shader when it is available, even with nothing to correct: it is the only
        // path that writes an OPAQUE alpha. A plain copy carries the game backbuffer alpha
        // through untouched, and a D3D9 backbuffer alpha is typically zero - SteamVR honours
        // that and composites the whole game as a see-through overlay on its own environment.
        if (ShaderBlitReady()) {
            ok = GammaDraw(g_uploadTex, g_w, g_h, srcX, g_srcEyeW, g_srcEyeH,
                           clampLoPx, clampHiPx, imgs[idx], g_eyeW, g_eyeH);
        }
        if (!ok) {
            // Degrade locally, never silently: drop the shift for this eye rather than failing
            // the whole submit (a failed eye used to collapse the entire frame to mono).
            if (shifted) {
                static bool warned = false;
                if (!warned) { warned = true;
                    XLOG("[OLVR][XR] convergence shift needs the shader blit - unavailable, "
                         "copying the unshifted window instead"); }
            }
            if (g_eyeW == g_srcEyeW && g_eyeH == g_srcEyeH) {
                const UINT x0 = (UINT)clampLoPx;
                D3D11_BOX box = { x0, 0, 0, x0 + g_srcEyeW, g_srcEyeH, 1 };
                g_ctx->CopySubresourceRegion(imgs[idx], 0, 0, 0, 0, g_uploadTex, 0, &box);
                ok = true;
            }
        }
    }
    XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
    g_fn.releaseSwapchainImage(sc, &ri);
    return ok;
}

// [LINKPOSE] (Beta 1.1) A projection layer's declared pose has to be where the head REALLY was, in the
// layer's own space. The pose that travels with each image is in the recenter frame (the yaw and
// position baselines - the frame the camera renders in), while the layer space is the runtime's raw
// LOCAL. Virtual Desktop shrugged the difference off; Meta's runtime (Quest Link / Air Link) shows
// every frame exactly at its declared pose, so after any recenter - the automatic one at start too,
// taken while the headset may still be lying on the desk - the picture sat that far away from where
// the player looks: "screens in a void that move to wherever I looked when I recentered", "the
// camera perspective below me". Mapping the pose back through the baseline makes declared == true
// on every runtime. [XR] DeclareTruePose=0 restores the old declaration.
static XrPosef RecenterToLocal(const XrPosef& rel) {
    if (!g_declLocal) return rel;
    XrPosef p;
    p.orientation = QNormalize(QMul(g_baseQ, rel.orientation));
    const XrVector3f r = QRotate(g_baseQ, rel.position);
    p.position = { g_baseP.x + r.x, g_baseP.y + r.y, g_baseP.z + r.z };
    return p;
}

// [FITFRUSTUM] Place one eye's rendered picture at its TRUE angular position inside an image
// that represents the RUNTIME's own frustum, black everywhere else.
//
// Why this exists: the Oculus/Quest Link runtime composites using its own per-eye frustum and
// ignores a differing declared FOV (ME1's [LINKFOV] finding, re-measured here 2026-08-05:
// rtFov e0 -54.0/+40.0 against this mod's symmetric +/-44). Declaring an intersection cannot help
// when the declaration is discarded - the submitted rect just gets stretched across the runtime's
// frustum instead, which displaces each eye ~5 deg in OPPOSITE directions and is exactly the
// unfusable doubling. Here the submitted image already spans the runtime's frustum, so the
// result is right whether the runtime honours the declaration or throws it away.
//
// srcX0/srcX1 bound this eye's half of the backbuffer (already carrying the convergence shift),
// ourFovX/Y are the angles that half actually covers, and rt is the eye the runtime wants.
// [LENSFIT] (Test36) the same with the source frustum given as tangents, so an off-centre render
// (each half drawn with its eye's own asymmetric frustum) maps 1:1 onto that eye.
static bool BlitFitToEyeTan(XrSwapchain sc, std::vector<ID3D11Texture2D*>& imgs,
                            float srcX0, float srcX1, float clampLoPx, float clampHiPx,
                            float tanOL, float tanOR, float tanOU, float tanOD, const XrFovf& rt);
static bool BlitFitToEye(XrSwapchain sc, std::vector<ID3D11Texture2D*>& imgs,
                         float srcX0, float srcX1, float clampLoPx, float clampHiPx,
                         float ourFovX, float ourFovY, const XrFovf& rt) {
    return BlitFitToEyeTan(sc, imgs, srcX0, srcX1, clampLoPx, clampHiPx,
                           tanf(-ourFovX * 0.5f), tanf(ourFovX * 0.5f),
                           tanf(ourFovY * 0.5f), tanf(-ourFovY * 0.5f), rt);
}
static bool BlitFitToEyeTan(XrSwapchain sc, std::vector<ID3D11Texture2D*>& imgs,
                            float srcX0, float srcX1, float clampLoPx, float clampHiPx,
                            float tanOL, float tanOR, float tanOU, float tanOD, const XrFovf& rt) {
    if (!sc || imgs.empty()) return false;
    const float tanTL = tanf(rt.angleLeft), tanTR = tanf(rt.angleRight);
    const float tanTU = tanf(rt.angleUp),   tanTD = tanf(rt.angleDown);
    const float spanTH = tanTR - tanTL, spanTV = tanTU - tanTD;
    const float spanOH = tanOR - tanOL, spanOV = tanOU - tanOD;
    if (spanTH < 1e-4f || spanTV < 1e-4f || spanOH < 1e-4f || spanOV < 1e-4f) return false;
    // The overlap of the rendered picture with what the eye covers.
    const float iL = (tanOL > tanTL) ? tanOL : tanTL, iR = (tanOR < tanTR) ? tanOR : tanTR;
    const float iU = (tanOU < tanTU) ? tanOU : tanTU, iD = (tanOD > tanTD) ? tanOD : tanTD;
    if (iR - iL < 1e-4f || iU - iD < 1e-4f) return false;
    // Where that overlap lands in the runtime's eye image...
    float dx0 = (iL - tanTL) / spanTH * (float)g_eyeW, dx1 = (iR - tanTL) / spanTH * (float)g_eyeW;
    float dy0 = (tanTU - iU) / spanTV * (float)g_eyeH, dy1 = (tanTU - iD) / spanTV * (float)g_eyeH;
    // ...and which part of the source picture it comes from.
    const float srcW = srcX1 - srcX0;
    const float sx0 = srcX0 + (iL - tanOL) / spanOH * srcW;
    const float sx1 = srcX0 + (iR - tanOL) / spanOH * srcW;
    const float sy0 = (tanOU - iU) / spanOV * (float)g_h;
    const float sy1 = (tanOU - iD) / spanOV * (float)g_h;
    if (dx1 - dx0 < 1.0f || dy1 - dy0 < 1.0f || sx1 - sx0 < 1.0f || sy1 - sy0 < 1.0f) return false;

    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (!XrSucceeded(g_fn.acquireSwapchainImage(sc, &ai, &idx))) return false;
    bool ok = false;
    XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
    if (XrSucceeded(g_fn.waitSwapchainImage(sc, &wi)) && ShaderBlitReady()) {
        ok = GammaDraw(g_uploadTex, g_w, g_h, sx0, (UINT)(sx1 - sx0), (UINT)(sy1 - sy0),
                       clampLoPx, clampHiPx, imgs[idx], g_eyeW, g_eyeH,
                       sy0, dx0, dy0, dx1 - dx0, dy1 - dy0);
    }
    XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
    g_fn.releaseSwapchainImage(sc, &ri);
    return ok;
}

// [STEAMVRSCENE] Build the black placeholder once, on SteamVR only. Every image in the chain is
// cleared to opaque black at creation, and the layer is re-submitted without ever being touched
// again, so this costs one 64x64 swapchain and nothing per frame.
static bool EnsureBlackScene() {
    if (g_blackSc) return true;
    if (g_blackTried) return false;
    g_blackTried = true;
    XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    ci.sampleCount = 1; ci.width = kBlackSceneSize; ci.height = kBlackSceneSize;
    ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
    if (!XrSucceeded(g_fn.createSwapchain(g_session, &ci, &g_blackSc))) {
        g_blackSc = 0; XLOG("[OLVR][XRRUNTIME] black scene swapchain create FAILED"); return false;
    }
    uint32_t n = 0; g_fn.enumerateSwapchainImages(g_blackSc, 0, &n, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> imgs(n);
    for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
    if (!XrSucceeded(g_fn.enumerateSwapchainImages(g_blackSc, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) {
        g_blackSc = 0; return false;
    }
    const float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    for (auto& im : imgs) {
        g_blackImgs.push_back(im.texture);
        ID3D11RenderTargetView* rtv = nullptr;
        if (SUCCEEDED(g_d3d11->CreateRenderTargetView(im.texture, nullptr, &rtv)) && rtv) {
            g_ctx->ClearRenderTargetView(rtv, black);
            rtv->Release();
        }
    }
    // One acquire/release cycle so the runtime treats the chain as having valid content.
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (XrSucceeded(g_fn.acquireSwapchainImage(g_blackSc, &ai, &idx))) {
        XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE;
        wi.timeout = XR_INFINITE_DURATION_VALUE;
        g_fn.waitSwapchainImage(g_blackSc, &wi);
        XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
        g_fn.releaseSwapchainImage(g_blackSc, &ri);
    }
    XLOG("[OLVR][XRRUNTIME] black scene layer ready (%d px: SteamVR scene keeper and snap-turn blink)", kBlackSceneSize);
    return true;
}

static bool LoadBrandLogo(std::vector<uint8_t>& pixels, UINT& width, UINT& height) {
    wchar_t path[MAX_PATH] = {};
    if (!GetModuleFileNameW(nullptr, path, MAX_PATH)) return false;
    wchar_t* slash = wcsrchr(path, L'\\');
    if (!slash) return false;
    wcscpy_s(slash + 1, MAX_PATH - (size_t)(slash + 1 - path), L"strikezone_logo.jpg");
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(init) && init != RPC_E_CHANGED_MODE) return false;
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapScaler* scaler = nullptr;
    IWICFormatConverter* converter = nullptr;
    bool ok = false;
    const UINT target = 144;
    if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_PPV_ARGS(&factory))) && factory &&
        SUCCEEDED(factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ,
                                                     WICDecodeMetadataCacheOnLoad, &decoder)) && decoder &&
        SUCCEEDED(decoder->GetFrame(0, &frame)) && frame &&
        SUCCEEDED(factory->CreateBitmapScaler(&scaler)) && scaler &&
        SUCCEEDED(scaler->Initialize(frame, target, target, WICBitmapInterpolationModeFant)) &&
        SUCCEEDED(factory->CreateFormatConverter(&converter)) && converter &&
        SUCCEEDED(converter->Initialize(scaler, GUID_WICPixelFormat32bppBGRA,
                                        WICBitmapDitherTypeNone, nullptr, 0.0,
                                        WICBitmapPaletteTypeCustom))) {
        width = target; height = target; pixels.resize((size_t)target * target * 4);
        ok = SUCCEEDED(converter->CopyPixels(nullptr, target * 4, (UINT)pixels.size(), pixels.data()));
    }
    if (converter) converter->Release();
    if (scaler) scaler->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (!ok) XLOG("[OLVR][CINEMA] strikezone_logo.jpg could not be loaded");
    return ok;
}

static bool BuildBrandPixels(std::vector<uint8_t>& pixels) {
    pixels.resize((size_t)kBrandW * kBrandH * 4);
    // Muted version of the supplied light blue. It keeps the same identity without becoming a
    // bright field inside the headset. The footer is darker for comfortable contrast.
    const uint8_t bgR = 58, bgG = 88, bgB = 112;
    const uint8_t bandR = 20, bandG = 27, bandB = 34;   // [THEATER] also the plaque under the screen
    const int footerTop = 830;
    for (int y = 0; y < kBrandH; ++y) {
        const bool band = y >= footerTop;
        for (int x = 0; x < kBrandW; ++x) {
            uint8_t* p = &pixels[((size_t)y * kBrandW + x) * 4];
            const float nx = fabsf((2.0f * x / (float)(kBrandW - 1)) - 1.0f);
            const float ny = fabsf((2.0f * y / (float)(kBrandH - 1)) - 1.0f);
            const float dim = band ? 1.0f : (1.0f - 0.12f * (nx * nx + ny * ny) * 0.5f);
            p[0] = (uint8_t)((band ? bandB : bgB) * dim);
            p[1] = (uint8_t)((band ? bandG : bgG) * dim);
            p[2] = (uint8_t)((band ? bandR : bgR) * dim);
            p[3] = 255;
        }
    }

    UINT lw = 0, lh = 0; std::vector<uint8_t> logo;
    if (LoadBrandLogo(logo, lw, lh)) {
        const int dx = 300, dy = footerTop + 24;
        for (UINT y = 0; y < lh && dy + (int)y < kBrandH; ++y) {
            for (UINT x = 0; x < lw && dx + (int)x < kBrandW; ++x) {
                const uint8_t* s = &logo[((size_t)y * lw + x) * 4];
                uint8_t* d = &pixels[((size_t)(dy + y) * kBrandW + dx + x) * 4];
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
            }
        }
    }

    BITMAPINFO bi = {}; bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = kBrandW; bi.bmiHeader.biHeight = -kBrandH;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    void* dibBits = nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    HBITMAP bmp = dc ? CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &dibBits, nullptr, 0) : nullptr;
    if (dc && bmp && dibBits) {
        memcpy(dibBits, pixels.data(), pixels.size());
        HGDIOBJ oldBmp = SelectObject(dc, bmp);
        HFONT font = CreateFontW(58, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                 OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
                                 DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
        HGDIOBJ oldFont = font ? SelectObject(dc, font) : nullptr;
        SetBkMode(dc, TRANSPARENT); SetTextColor(dc, RGB(240, 245, 248));
        RECT textRect = { 480, footerTop + 42, 1450, kBrandH - 20 };
        DrawTextW(dc, L"VR mod made by StrikeZoneVR", -1, &textRect,
                  DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
        if (oldFont) SelectObject(dc, oldFont);
        if (font) DeleteObject(font);
        SelectObject(dc, oldBmp);
        memcpy(pixels.data(), dibBits, pixels.size());
        DeleteObject(bmp); DeleteDC(dc);
    } else {
        if (bmp) DeleteObject(bmp);
        if (dc) DeleteDC(dc);
        XLOG("[OLVR][CINEMA] footer text creation failed");
    }
    return true;
}

static bool EnsureBrandScene() {
    if (!g_brandOn) return false;
    if (g_brandReady && g_brandSc) return true;
    if (g_brandTried) return false;
    g_brandTried = true;
    XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    ci.sampleCount = 1; ci.width = kBrandW; ci.height = kBrandH;
    ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
    if (!XrSucceeded(g_fn.createSwapchain(g_session, &ci, &g_brandSc))) {
        g_brandSc = 0; XLOG("[OLVR][CINEMA] branded backdrop swapchain creation failed"); return false;
    }
    uint32_t n = 0; g_fn.enumerateSwapchainImages(g_brandSc, 0, &n, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> imgs(n);
    for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
    if (!XrSucceeded(g_fn.enumerateSwapchainImages(g_brandSc, n, &n,
                        (XrSwapchainImageBaseHeader*)imgs.data()))) return false;
    std::vector<uint8_t> pixels; BuildBrandPixels(pixels);
    for (auto& im : imgs) {
        g_brandImgs.push_back(im.texture);
        g_ctx->UpdateSubresource(im.texture, 0, nullptr, pixels.data(), kBrandW * 4, 0);
    }
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (XrSucceeded(g_fn.acquireSwapchainImage(g_brandSc, &ai, &idx))) {
        XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
        g_fn.waitSwapchainImage(g_brandSc, &wi);
        XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
        g_fn.releaseSwapchainImage(g_brandSc, &ri);
    }
    g_brandReady = true;
    XLOG("[OLVR][CINEMA] muted-blue StrikeZoneVR backdrop ready");
    return true;
}

// ---- [MARKERS] (Test37) developer markers for the physical interactions ---------------------------
// Rings drawn by the compositor on top of the game: where a door knob can be grabbed, the battery
// pouch, the raise-to-eye zone, the battery in your hand. Positions arrive from the game thread
// (ol_interact.cpp) in the recenter frame; each frame they become billboard quads in LOCAL space.
// Developer aid only ([Motion] DevMarkers).
static const int kMarkerTex = 64, kMaxMarkers = 16;
static const int kLabelW = 512, kLabelH = 160;
// (Beta 1.1) [BATGAUGE] five battery pictures (0..4 bars) and the digits 0..9, in two rows under the labels
static const int kBatW = 96, kBatH = 48, kDigW = 32, kDigH = 48;
static int g_batRect[5][4] = {}, g_digRect[10][4] = {};
extern "C" int OLInteract_BatteryGauge(float* energy, int* spares);
static XrSwapchain g_mkAtlas = 0;                    // every ring, glow and label in one image
static int g_mkRect[OLMK_COUNT][4] = {};              // x, y, w, h inside it
static bool g_mkTried = false, g_mkReady = false;
static OLMarker g_mk[kMaxMarkers];
static int g_mkN = 0;
static SRWLOCK g_mkLock = SRWLOCK_INIT;
extern "C" void OLXR_SetMarkers(const OLMarker* m, int n) {
    if (n < 0) n = 0; if (n > kMaxMarkers) n = kMaxMarkers;
    AcquireSRWLockExclusive(&g_mkLock);
    if (n && m) memcpy(g_mk, m, sizeof(OLMarker) * n);
    g_mkN = n;
    ReleaseSRWLockExclusive(&g_mkLock);
}
// [HOWTO] (Test38) A how-to label: the camcorder's on-screen-display look - a dark glass panel with a
// thin border, a bold title, a smaller hint line, and an arrow under it pointing at the spot. Text is
// drawn white-on-black by GDI and its coverage becomes alpha; the panel and arrow are analytic
// (anti-aliased by distance). Unpremultiplied sRGB BGRA, kLabelW x kLabelH.
static void MakeLabel(const wchar_t* title, const wchar_t* hint, bool warn, std::vector<uint8_t>& out) {
    const int W = kLabelW, H = kLabelH, panelH = 124, arrowH = H - panelH;
    out.assign((size_t)W * H * 4, 0);
    std::vector<uint8_t> cov((size_t)W * H * 2, 0);   // [0] title coverage, [1] hint coverage
    HDC dc = CreateCompatibleDC(nullptr);
    BITMAPINFO bi = {}; bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = W; bi.bmiHeader.biHeight = -H; bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = dc ? CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0) : nullptr;
    if (dc && bmp && bits) {
        HGDIOBJ old = SelectObject(dc, bmp);
        SetBkMode(dc, TRANSPARENT); SetTextColor(dc, RGB(255, 255, 255));
        for (int pass = 0; pass < 2; ++pass) {
            memset(bits, 0, (size_t)W * H * 4);
            HFONT f = CreateFontW(pass ? -30 : -50, 0, 0, 0, pass ? FW_NORMAL : FW_BOLD, 0, 0, 0, DEFAULT_CHARSET,
                                  OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN, L"Consolas");
            HGDIOBJ of = SelectObject(dc, f);
            RECT r = { 12, pass ? 72 : 10, W - 12, pass ? 116 : 72 };
            DrawTextW(dc, pass ? hint : title, -1, &r, DT_CENTER | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX);
            GdiFlush();
            const uint8_t* b = (const uint8_t*)bits;
            for (int i = 0; i < W * H; ++i) cov[(size_t)i * 2 + pass] = b[(size_t)i * 4 + 1];
            SelectObject(dc, of); DeleteObject(f);
        }
        SelectObject(dc, old);
    }
    if (bmp) DeleteObject(bmp);
    if (dc) DeleteDC(dc);
    // colours (B,G,R): accent white / warning red; hint soft green / soft red; panel near-black
    const float acc[3] = { warn ? 70.0f : 235.0f, warn ? 80.0f : 240.0f, warn ? 255.0f : 235.0f };
    const float hintC[3] = { warn ? 150.0f : 150.0f, warn ? 160.0f : 235.0f, warn ? 255.0f : 150.0f };
    const float rad = 18.0f, bw = 2.5f;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const float px = x + 0.5f, py = y + 0.5f;
        float a = 0.0f, c[3] = { 14.0f, 16.0f, 14.0f };
        if (py < panelH) {
            // rounded-rectangle signed distance (negative inside)
            const float hx = W * 0.5f - 3.0f, hy = panelH * 0.5f - 3.0f;
            const float qx = fabsf(px - W * 0.5f) - (hx - rad), qy = fabsf(py - panelH * 0.5f) - (hy - rad);
            const float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
            const float d = sqrtf(ox * ox + oy * oy) + (qx > qy ? (qx < 0 ? qx : 0) : (qy < 0 ? qy : 0)) - rad;
            const float inside = d < -0.5f ? 1.0f : (d > 0.5f ? 0.0f : 0.5f - d);
            const float border = 1.0f - (fabsf(d + bw * 0.5f) - bw * 0.5f);
            const float bA = border < 0 ? 0 : (border > 1 ? 1 : border);
            a = inside * 0.74f;
            for (int k = 0; k < 3; ++k) c[k] = c[k] + (acc[k] - c[k]) * bA;
            if (bA * inside > 0.0f) a = a + (0.95f - a) * bA;
        } else {
            // the arrow: a filled triangle pointing down, centred
            const float t = (py - panelH) / (float)arrowH;           // 0 at the panel, 1 at the tip
            const float half = 22.0f * (1.0f - t);
            const float dx = fabsf(px - W * 0.5f);
            const float e = half - dx;
            a = e > 0.5f ? 0.95f : (e < -0.5f ? 0.0f : (e + 0.5f) * 0.95f);
            for (int k = 0; k < 3; ++k) c[k] = acc[k];
        }
        const float ct = cov[((size_t)y * W + x) * 2] / 255.0f, ch = cov[((size_t)y * W + x) * 2 + 1] / 255.0f;
        for (int k = 0; k < 3; ++k) c[k] = c[k] + (acc[k] - c[k]) * ct;
        for (int k = 0; k < 3; ++k) c[k] = c[k] + (hintC[k] - c[k]) * ch;
        if (ct > 0 || ch > 0) { const float tc = ct > ch ? ct : ch; a = a + (1.0f - a) * tc; }
        uint8_t* p = &out[((size_t)y * W + x) * 4];
        for (int k = 0; k < 3; ++k) p[k] = (uint8_t)(c[k] < 0 ? 0 : (c[k] > 255 ? 255 : c[k] + 0.5f));
        p[3] = (uint8_t)(a < 0 ? 0 : (a > 1 ? 255 : a * 255.0f + 0.5f));
    }
}
static bool EnsureMarkers() {
    if (g_mkReady) return true;
    if (g_mkTried) return false;
    g_mkTried = true;
    // sRGB-encoded B,G,R per ring colour, then the two glows
    static const uint8_t kCol[OLMK_GLOW_RED + 1][3] = {
        { 90, 255, 70 }, { 60, 60, 255 }, { 40, 230, 255 }, { 255, 230, 60 }, { 30, 150, 255 }, { 255, 255, 255 },
        { 170, 255, 175 }, { 70, 90, 255 } };
    struct LabelText { const wchar_t* title; const wchar_t* hint; bool warn; };
    static const LabelText kLabels[OLMK_COUNT - OLMK_L_TAKE_CAM] = {
        { L"TAKE CAMCORDER",     L"squeeze the grip",           false },
        { L"LOOK THROUGH IT",    L"bring it up to your eye",    false },
        { L"PULL OUT BATTERY",   L"left grip, pull it away",    false },
        { L"DROP IT",            L"open your hand",             false },
        { L"NEW BATTERY",        L"left hip, squeeze grip",     false },
        { L"SLIDE IT IN",        L"touch it to the camcorder",  false },
        { L"NO SPARE BATTERIES", L"find more in the asylum",    true  },
        { L"OPEN THE DOOR",      L"push it, or grab the handle", false },
        { L"LOCKED",             L"find another way",           true  },
        { L"BATTERY CHARGED",    L"",                           false },
        { L"CLIMB",              L"grab the edge, pull down",   false },
        { L"PULL YOURSELF UP",   L"both grips, pull down",      false },
        { L"COVER YOUR MOUTH",   L"hand over it: hold your breath", false },
        { L"RECENTER",           L"click both thumbsticks",     false } };
    // One atlas for everything (Test38: 18 separate swapchains were more than SteamVR would create):
    // the rings and glows side by side in the top 64-pixel row, the labels stacked below it.
    // (Beta 1.1) [BATGAUGE] then a row of battery pictures and a row of digits.
    const int labelsEnd = kMarkerTex + (OLMK_COUNT - OLMK_L_TAKE_CAM) * kLabelH;
    const int AW = kLabelW, AH = labelsEnd + kBatH + kDigH;
    std::vector<uint8_t> atlas((size_t)AW * AH * 4, 0), px;
    for (int c = 0; c < OLMK_COUNT; ++c) {
        int tw = kMarkerTex, th = kMarkerTex, ox = 0, oy = 0;
        if (c >= OLMK_L_TAKE_CAM) {
            const LabelText& lt = kLabels[c - OLMK_L_TAKE_CAM];
            MakeLabel(lt.title, lt.hint, lt.warn, px); tw = kLabelW; th = kLabelH;
            ox = 0; oy = kMarkerTex + (c - OLMK_L_TAKE_CAM) * kLabelH;
        } else {
            ox = c * kMarkerTex; oy = 0;
            px.assign((size_t)kMarkerTex * kMarkerTex * 4, 0);
            for (int y = 0; y < kMarkerTex; ++y) for (int x = 0; x < kMarkerTex; ++x) {
                const float fx = (x + 0.5f) / kMarkerTex * 2.0f - 1.0f, fy = (y + 0.5f) / kMarkerTex * 2.0f - 1.0f;
                const float r = sqrtf(fx * fx + fy * fy);
                float a;
                if (c >= OLMK_GLOW) {
                    // [HOWTO] a soft glow: a gentle disc with a brighter thin rim, no hard edge
                    const float fill = r < 1.0f ? 0.45f * (1.0f - r) * (1.0f - r) : 0.0f;
                    float rim = 1.0f - fabsf(r - 0.78f) / 0.07f; rim = rim < 0 ? 0 : rim * 0.55f;
                    a = fill > rim ? fill : rim;
                } else {
                    a = 1.0f - fabsf(r - 0.80f) / 0.10f;                 // ring band
                    if (r < 0.20f) a = 1.0f;
                    else if (r < 0.26f) { const float d = 1.0f - (r - 0.20f) / 0.06f; if (d > a) a = d; }
                    const float glow = (r < 1.0f) ? 0.22f * (1.0f - r) : 0.0f;  // faint fill
                    if (glow > a) a = glow;
                }
                if (a < 0.0f) a = 0.0f; if (a > 1.0f) a = 1.0f;
                uint8_t* p = &px[((size_t)y * kMarkerTex + x) * 4];
                p[0] = kCol[c][0]; p[1] = kCol[c][1]; p[2] = kCol[c][2]; p[3] = (uint8_t)(a * 255.0f + 0.5f);
            }
        }
        for (int y = 0; y < th; ++y) memcpy(&atlas[(((size_t)(oy + y)) * AW + ox) * 4], &px[(size_t)y * tw * 4], (size_t)tw * 4);
        g_mkRect[c][0] = ox; g_mkRect[c][1] = oy; g_mkRect[c][2] = tw; g_mkRect[c][3] = th;
    }
    {   // (Beta 1.1) [BATGAUGE] the camcorder's battery: an outlined cell with its terminal, 0..4 bars inside
        // (white; red when one bar or none is left). Then 7-segment digits. No plate behind either: the
        // gauge floats over the game with nothing but its own white lines, so there is no dark rectangle
        // hanging low in the view where the game's picture should be.
        auto put = [&](int x, int y, float b, float g, float r, float a) {
            if (x < 0 || y < 0 || x >= AW || y >= AH) return;
            uint8_t* p = &atlas[((size_t)y * AW + x) * 4];
            p[0] = (uint8_t)b; p[1] = (uint8_t)g; p[2] = (uint8_t)r; p[3] = (uint8_t)(a * 255.0f + 0.5f);
        };
        for (int s = 0; s < 5; ++s) {
            const int ox = s * kBatW, oy = labelsEnd;
            const bool low = s <= 1;
            for (int y = 0; y < kBatH; ++y) for (int x = 0; x < kBatW; ++x) {
                float a = 0.0f, cb = 0, cg = 0, cr = 0;                             // nothing behind the cell
                const bool body = x >= 8 && x < kBatW - 16 && y >= 8 && y < kBatH - 8;
                const bool bodyIn = x >= 12 && x < kBatW - 20 && y >= 12 && y < kBatH - 12;
                const bool nub = x >= kBatW - 16 && x < kBatW - 9 && y >= 17 && y < kBatH - 17;
                if ((body && !bodyIn) || nub) { cb = cg = cr = 240; a = 1.0f; }
                if (bodyIn) {
                    const int inner = (kBatW - 20) - 12 - 4;                        // 4 bars with 4 px gaps
                    const int bw = (inner - 3 * 4) / 4;
                    const int rel = x - 14;
                    const int bar = rel / (bw + 4), within = rel % (bw + 4);
                    if (y >= 14 && y < kBatH - 14 && rel >= 0 && bar < s && within < bw) {
                        if (low) { cb = 60; cg = 60; cr = 255; } else { cb = 235; cg = 245; cr = 235; }
                        a = 1.0f;
                    }
                }
                put(ox + x, oy + y, cb, cg, cr, a);
            }
            g_batRect[s][0] = ox; g_batRect[s][1] = oy; g_batRect[s][2] = kBatW; g_batRect[s][3] = kBatH;
        }
        // 7 segments: a top, b top-right, c bottom-right, d bottom, e bottom-left, f top-left, g middle
        static const uint8_t kSeg[10] = { 0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F };
        for (int d = 0; d < 10; ++d) {
            const int ox = d * kDigW, oy = labelsEnd + kBatH;
            for (int y = 0; y < kDigH; ++y) for (int x = 0; x < kDigW; ++x) {
                const int L = 7, Rr = kDigW - 8, T = 7, M = kDigH / 2, B = kDigH - 8, t = 2;
                bool on = false;
                const uint8_t m = kSeg[d];
                if ((m & 0x01) && y >= T - t && y <= T + t && x >= L && x <= Rr) on = true;
                if ((m & 0x02) && x >= Rr - t && x <= Rr + t && y >= T && y <= M) on = true;
                if ((m & 0x04) && x >= Rr - t && x <= Rr + t && y >= M && y <= B) on = true;
                if ((m & 0x08) && y >= B - t && y <= B + t && x >= L && x <= Rr) on = true;
                if ((m & 0x10) && x >= L - t && x <= L + t && y >= M && y <= B) on = true;
                if ((m & 0x20) && x >= L - t && x <= L + t && y >= T && y <= M) on = true;
                if ((m & 0x40) && y >= M - t && y <= M + t && x >= L && x <= Rr) on = true;
                if (on) put(ox + x, oy + y, 240, 240, 240, 1.0f);
                else    put(ox + x, oy + y, 0, 0, 0, 0.0f);                          // lit segments only
            }
            g_digRect[d][0] = ox; g_digRect[d][1] = oy; g_digRect[d][2] = kDigW; g_digRect[d][3] = kDigH;
        }
    }
    XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
    ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    ci.sampleCount = 1; ci.width = AW; ci.height = AH;
    ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
    const XrResult cr = g_fn.createSwapchain(g_session, &ci, &g_mkAtlas);
    if (!XrSucceeded(cr)) {
        g_mkAtlas = 0; XLOG("[OLVR][MARKERS] atlas swapchain %dx%d creation failed (%d) - markers off", AW, AH, (int)cr); return false;
    }
    uint32_t n = 0; g_fn.enumerateSwapchainImages(g_mkAtlas, 0, &n, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> imgs(n);
    for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
    if (!XrSucceeded(g_fn.enumerateSwapchainImages(g_mkAtlas, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) return false;
    for (auto& im : imgs) g_ctx->UpdateSubresource(im.texture, 0, nullptr, atlas.data(), AW * 4, 0);
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo acq = {}; acq.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (XrSucceeded(g_fn.acquireSwapchainImage(g_mkAtlas, &acq, &idx))) {
        XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
        g_fn.waitSwapchainImage(g_mkAtlas, &wi);
        XrSwapchainImageReleaseInfo rel = {}; rel.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
        g_fn.releaseSwapchainImage(g_mkAtlas, &rel);
    }
    g_mkReady = true;
    XLOG("[OLVR][MARKERS] markers ready: 6 rings, 2 glows, %d how-to labels", (int)(OLMK_COUNT - OLMK_L_TAKE_CAM));
    return true;
}

// ---- [FLIPSCREEN] (Test43) the camcorder's flip-out screen, live ---------------------------------
// The free-hold camcorder (Test43) shows what its lens sees on its flip-out screen. There is no second
// camera to render from, so the screen shows the part of this frame's left-eye picture the lens points
// at - the lens direction (from the camcorder in Miles's hand) projected into the eye the frame was
// rendered for, cropped to a camcorder's ~50 degrees - with the camcorder's night-vision look. Pointed
// away from everything in view it shows static. Drawn every compositor frame into its own small image
// (never straight into a swapchain image - see the blit's note above) and shown as a quad over the
// model's screen.
static const UINT kFsW = 384, kFsH = 288;
static XrSwapchain g_fsChain = 0;
static std::vector<ID3D11Texture2D*> g_fsImgs;
static ID3D11Texture2D* g_fsTmp = nullptr; static ID3D11RenderTargetView* g_fsRtv = nullptr;
static ID3D11VertexShader* g_fsVS = nullptr; static ID3D11PixelShader* g_fsPS = nullptr; static ID3D11Buffer* g_fsCB = nullptr;
static bool g_fsTried = false, g_fsReady = false;
static OLFlipScreen g_fs = {}; static volatile ULONGLONG g_fsMs = 0; static SRWLOCK g_fsLock = SRWLOCK_INIT;
// how the stereo frame on screen was rendered (set by the SBS path)
static bool g_fsLens = false; static float g_fsFx = 0.0f, g_fsFy = 0.0f; static XrQuaternionf g_fsViewQ = { 0, 0, 0, 1 };
extern "C" void OLXR_SetFlipScreen(const OLFlipScreen* s) {
    AcquireSRWLockExclusive(&g_fsLock);
    if (s) g_fs = *s; else g_fs.on = 0;
    g_fsMs = GetTickCount64();
    ReleaseSRWLockExclusive(&g_fsLock);
}
static const char* kFlipHlsl =
"cbuffer CB : register(b0) { float4 box; float4 lim; float4 tune; };\n"   // u0,v0,du,dv | u/v limits | time, onImage, gain
"struct VOut { float4 p : SV_Position; float2 uv : TEXCOORD0; float2 q : TEXCOORD1; };\n"
"VOut vs(uint id : SV_VertexID) {\n"
"  float2 t = float2((id << 1) & 2, id & 2);\n"
"  VOut o; o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"  o.uv = box.xy + t * box.zw; o.q = t; return o; }\n"
"Texture2D srcT : register(t0);\n"
"SamplerState s0 : register(s0);\n"
"float H(float2 p) { return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453); }\n"
"float4 ps(VOut i) : SV_Target {\n"
"  float n = H(floor(i.q * float2(192, 144)) + tune.x * 61.0);\n"
"  float3 c;\n"
"  if (tune.y > 0.5 && i.uv.x >= lim.x && i.uv.x <= lim.y && i.uv.y >= lim.z && i.uv.y <= lim.w) {\n"
"    float3 s = srcT.SampleLevel(s0, i.uv, 0).rgb;\n"                    // gamma-space bytes
"    if (tune.w > 0.5) {\n"                                                // night vision: amplified low light, green, grainy
"      float l = pow(saturate(dot(s, float3(0.30, 0.59, 0.11)) * tune.z), 0.5);\n"
"      c = l * float3(0.42, 1.0, 0.48) + (n - 0.5) * 0.13;\n"
"    } else {\n"                                                           // colour, as a camcorder screen shows it
"      float3 g = pow(saturate(s * 1.5), 0.9);\n"
"      c = lerp(g, dot(g, float3(0.30, 0.59, 0.11)).xxx, 0.15) + (n - 0.5) * 0.04;\n"
"    }\n"
"  } else {\n"
"    c = (n * 0.55 + 0.05) * float3(0.80, 0.92, 0.80);\n"                // no picture: static
"  }\n"
"  float2 d = i.q - 0.5; c *= saturate(1.12 - dot(d, d) * 1.5);\n"       // vignette
"  c *= 0.92 + 0.08 * sin(i.q.y * 288.0 * 3.14159);\n"                    // scan lines
"  if (frac(tune.x * 0.5) < 0.6 && length((i.q - float2(0.09, 0.12)) * float2(1.33, 1.0)) < 0.035) c = float3(1.0, 0.12, 0.08);\n"
"  return float4(pow(saturate(c), 2.2), 1.0); }\n";                    // the target is sRGB
static bool EnsureFlip() {
    if (g_fsReady) return true;
    if (g_fsTried) return false;
    g_fsTried = true;
    if (!EnsureGammaPipeline()) return false;                         // its sampler and states
    HMODULE dc = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3DCompile compile = dc ? (PFN_D3DCompile)GetProcAddress(dc, "D3DCompile") : nullptr;
    ID3DBlob* vsb = nullptr; ID3DBlob* psb = nullptr; ID3DBlob* err = nullptr;
    if (!compile || FAILED(compile(kFlipHlsl, strlen(kFlipHlsl), nullptr, nullptr, nullptr, "vs", "vs_4_0", 0, 0, &vsb, &err)) ||
        FAILED(compile(kFlipHlsl, strlen(kFlipHlsl), nullptr, nullptr, nullptr, "ps", "ps_4_0", 0, 0, &psb, &err))) {
        XLOG("[OLVR][FLIPSCREEN] shader compile FAILED%s%s - no live flip screen", err ? ": " : "", err ? (const char*)err->GetBufferPointer() : "");
        if (vsb) vsb->Release(); if (psb) psb->Release(); if (err) err->Release();
        return false;
    }
    bool ok = SUCCEEDED(g_d3d11->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_fsVS)) &&
              SUCCEEDED(g_d3d11->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_fsPS));
    vsb->Release(); psb->Release();
    if (ok) { D3D11_BUFFER_DESC bd = {}; bd.ByteWidth = 48; bd.Usage = D3D11_USAGE_DEFAULT; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
              ok = SUCCEEDED(g_d3d11->CreateBuffer(&bd, nullptr, &g_fsCB)); }
    if (ok) {
        D3D11_TEXTURE2D_DESC td = {}; td.Width = kFsW; td.Height = kFsH; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
        ok = SUCCEEDED(g_d3d11->CreateTexture2D(&td, nullptr, &g_fsTmp)) && SUCCEEDED(g_d3d11->CreateRenderTargetView(g_fsTmp, nullptr, &g_fsRtv));
    }
    if (ok) {
        XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
        ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
        ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; ci.sampleCount = 1; ci.width = kFsW; ci.height = kFsH;
        ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
        ok = XrSucceeded(g_fn.createSwapchain(g_session, &ci, &g_fsChain));
        if (ok) {
            uint32_t n = 0; g_fn.enumerateSwapchainImages(g_fsChain, 0, &n, nullptr);
            std::vector<XrSwapchainImageD3D11KHR> imgs(n);
            for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
            ok = XrSucceeded(g_fn.enumerateSwapchainImages(g_fsChain, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()));
            if (ok) for (auto& im : imgs) g_fsImgs.push_back(im.texture);
        } else g_fsChain = 0;
    }
    g_fsReady = ok;
    XLOG("[OLVR][FLIPSCREEN] live camcorder screen %s (%ux%u)", ok ? "ready" : "FAILED - the model's own screen only", kFsW, kFsH);
    return ok;
}
// [LENSCAM] (Test46) The camcorder's own lens: the game renders the lens's view (a scene capture at the
// lens - ol_ue3.cpp [LIVESCREEN]) into a small render target, and the proxy knows that target's GPU
// texture by its size when the game makes it (OLProxy_LensTex). Each Present (game render thread) it is
// copied into one of three shared textures - D3D9Ex shared handles, as the frames are - and an event
// query says when the copy is done; the bridge then shows the newest finished copy on the flip screen
// instead of a crop of the eye picture. Point the camcorder behind you and the screen shows behind you.
extern "C" IDirect3DTexture9* OLProxy_LensTex();
extern "C" int OLUE3_LensLive();
struct LensSlot { IDirect3DTexture9* tex9; IDirect3DSurface9* surf9; IDirect3DQuery9* done9; HANDLE sh;
                  ID3D11Texture2D* tex11; ID3D11ShaderResourceView* srv; };
static LensSlot g_lens[3] = {};
static const UINT kLensW = 352, kLensH = 264;   // the capture's own size (ol_ue3.cpp kLsW / kLsH)
static volatile long g_lensPub = -1;            // newest finished copy
static volatile ULONGLONG g_lensPubMs = 0;
static bool g_lensFailed = false, g_lensLogged = false;
// Game render thread, at Present.
static void LensPublish(IDirect3DDevice9* dev) {
    static int s_pending = -1, s_next = 0;
    if (s_pending >= 0 && g_lens[s_pending].done9->GetData(nullptr, 0, 0) == S_OK) {
        InterlockedExchange(&g_lensPub, s_pending); g_lensPubMs = GetTickCount64(); s_pending = -1;
    }
    IDirect3DTexture9* src = OLProxy_LensTex();
    if (!src || g_lensFailed || s_pending >= 0 || !OLUE3_LensLive()) return;
    static ULONGLONG s_lastMs = 0;                 // the capture renders 30 times a second: copy no more often
    const ULONGLONG nowMs = GetTickCount64();
    if (nowMs - s_lastMs < 30) return;
    s_lastMs = nowMs;
    if (!g_lens[0].tex9) {
        for (auto& sl : g_lens) {
            HANDLE sh = nullptr; HRESULT hr;
            if (FAILED(hr = dev->CreateTexture(kLensW, kLensH, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &sl.tex9, &sh)) ||
                !sl.tex9 || !sh || FAILED(hr = sl.tex9->GetSurfaceLevel(0, &sl.surf9)) || FAILED(hr = dev->CreateQuery(D3DQUERYTYPE_EVENT, &sl.done9))) {
                g_lensFailed = true; XLOG("[OLVR][LENSCAM] shared lens frames FAILED hr=0x%08lX - the flip screen shows the eye picture", hr); return;
            }
            sl.sh = sh;
        }
        XLOG("[OLVR][LENSCAM] 3 shared lens frames %ux%u", kLensW, kLensH);
    }
    int k = s_next; if (k == g_lensPub) k = (k + 1) % 3;
    IDirect3DSurface9* ss = nullptr;
    if (SUCCEEDED(src->GetSurfaceLevel(0, &ss)) && ss) {
        if (SUCCEEDED(dev->StretchRect(ss, nullptr, g_lens[k].surf9, nullptr, D3DTEXF_LINEAR))) {
            g_lens[k].done9->Issue(D3DISSUE_END); s_pending = k; s_next = (k + 1) % 3;
        } else if (!g_lensLogged) { g_lensLogged = true; XLOG("[OLVR][LENSCAM] lens copy failed"); }
        ss->Release();
    }
}
// Bridge thread: the newest lens picture, or nullptr (then the flip screen crops the eye picture).
static ID3D11ShaderResourceView* LensSrv() {
    const long i = g_lensPub;
    if (i < 0 || i > 2 || GetTickCount64() - g_lensPubMs > 300 || !g_lens[i].sh) return nullptr;
    LensSlot& sl = g_lens[i];
    if (!sl.srv) {
        if (!sl.tex11 && (FAILED(g_d3d11->OpenSharedResource(sl.sh, __uuidof(ID3D11Texture2D), (void**)&sl.tex11)) || !sl.tex11)) {
            sl.tex11 = nullptr; sl.sh = nullptr; XLOG("[OLVR][LENSCAM] the bridge could not open a lens frame"); return nullptr;
        }
        if (FAILED(g_d3d11->CreateShaderResourceView(sl.tex11, nullptr, &sl.srv))) { sl.srv = nullptr; return nullptr; }
        static long s_n = 0;
        if (InterlockedIncrement(&s_n) == 1) XLOG("[OLVR][LENSCAM] the flip screen shows the camcorder's lens");
    }
    return sl.srv;
}

// Draws this frame's screen picture and fills `q`. false: nothing to show.
static bool FlipScreenLayer(XrCompositionLayerQuad& q) {
    OLFlipScreen fs;
    AcquireSRWLockShared(&g_fsLock); fs = g_fs; const ULONGLONG ms = g_fsMs; ReleaseSRWLockShared(&g_fsLock);
    if (!fs.on || !ms || GetTickCount64() - ms > 200 || !g_uploadTex || g_w < 2 || g_h < 2) return false;
    if (!EnsureFlip()) return false;
    // the lens, in the view the left eye was rendered from (-Z ahead, +X right, +Y up)
    const XrVector3f v = QRotate(QConj(g_fsViewQ), { fs.lens[0], fs.lens[1], fs.lens[2] });
    float tl, tr, tu, td;
    if (g_fsLens && g_rtEyeFovValid) { tl = tanf(g_rtEyeFov[0].angleLeft); tr = tanf(g_rtEyeFov[0].angleRight); tu = tanf(g_rtEyeFov[0].angleUp); td = tanf(g_rtEyeFov[0].angleDown); }
    else { tr = tanf(g_fsFx * 0.5f); tl = -tr; tu = tanf(g_fsFy * 0.5f); td = -tu; }
    const bool onImage = v.z < -0.25f && tr - tl > 0.01f && tu - td > 0.01f;
    const float tx = onImage ? v.x / -v.z : 0.0f, ty = onImage ? v.y / -v.z : 0.0f;
    const float hw = 0.466f, hh = 0.35f;                               // ~50 x 39 degrees, 4:3
    const float u0 = (tx - hw - tl) / (tr - tl), u1 = (tx + hw - tl) / (tr - tl);
    const float v0 = (tu - (ty + hh)) / (tu - td), v1 = (tu - (ty - hh)) / (tu - td);
    // the left eye is the left half of the uploaded frame
    const float t = (float)((GetTickCount64() % 100000ULL) * 0.001);
    float cb[12] = { u0 * 0.5f, v0, (u1 - u0) * 0.5f, v1 - v0,  0.0f, 0.5f, 0.0f, 1.0f,  t, onImage ? 1.0f : 0.0f, fs.nv ? 9.0f : 1.0f, fs.nv ? 1.0f : 0.0f };
    ID3D11ShaderResourceView* srv = LensSrv();                         // [LENSCAM] the lens itself, whole
    const bool lens = srv != nullptr;
    if (lens) { srv->AddRef(); const float lc[8] = { 0, 0, 1, 1,  0, 1, 0, 1 }; memcpy(cb, lc, sizeof(lc)); cb[9] = 1.0f; }
    else if (FAILED(g_d3d11->CreateShaderResourceView(g_uploadTex, nullptr, &srv))) return false;
    g_ctx->UpdateSubresource(g_fsCB, 0, nullptr, cb, 0, 0);
    const D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)kFsW, (float)kFsH, 0.0f, 1.0f };
    const float blendFactor[4] = { 0, 0, 0, 0 };
    g_ctx->IASetInputLayout(nullptr); g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_ctx->VSSetShader(g_fsVS, nullptr, 0); g_ctx->VSSetConstantBuffers(0, 1, &g_fsCB);
    g_ctx->PSSetShader(g_fsPS, nullptr, 0); g_ctx->PSSetConstantBuffers(0, 1, &g_fsCB);
    g_ctx->PSSetShaderResources(0, 1, &srv); g_ctx->PSSetSamplers(0, 1, &g_blitSamp);
    g_ctx->OMSetBlendState(g_blitBlend, blendFactor, 0xffffffff); g_ctx->OMSetDepthStencilState(g_blitDepth, 0);
    g_ctx->OMSetRenderTargets(1, &g_fsRtv, nullptr); g_ctx->RSSetState(g_blitRaster); g_ctx->RSSetViewports(1, &vp);
    g_ctx->Draw(3, 0);
    ID3D11ShaderResourceView* nul = nullptr; ID3D11RenderTargetView* nulrt = nullptr;
    g_ctx->PSSetShaderResources(0, 1, &nul); g_ctx->OMSetRenderTargets(1, &nulrt, nullptr);
    srv->Release();
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (!XrSucceeded(g_fn.acquireSwapchainImage(g_fsChain, &ai, &idx)) || idx >= g_fsImgs.size()) return false;
    XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
    const bool waited = XrSucceeded(g_fn.waitSwapchainImage(g_fsChain, &wi));
    if (waited) g_ctx->CopyResource(g_fsImgs[idx], g_fsTmp);
    XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
    g_fn.releaseSwapchainImage(g_fsChain, &ri);
    if (!waited) return false;
    const XrVector3f r = QRotate(g_baseQ, { fs.pos[0], fs.pos[1], fs.pos[2] });
    q.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE; q.next = nullptr; q.layerFlags = 0;
    q.space = g_localSpace; q.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
    q.subImage.swapchain = g_fsChain; q.subImage.imageArrayIndex = 0;
    q.subImage.imageRect.offset = { 0, 0 }; q.subImage.imageRect.extent = { (int32_t)kFsW, (int32_t)kFsH };
    q.pose.orientation = QNormalize(QMul(g_baseQ, { fs.q[0], fs.q[1], fs.q[2], fs.q[3] }));
    q.pose.position = { g_baseP.x + r.x, g_baseP.y + r.y, g_baseP.z + r.z };
    q.size.width = fs.w; q.size.height = fs.h;
    static long s_logs = 0;
    if (InterlockedIncrement(&s_logs) <= 2 || (s_logs % 2000) == 0)
        XLOG("[OLVR][FLIPSCREEN] screen %.3fx%.3f m, %s, %s (view %+.2f %+.2f %+.2f)", fs.w, fs.h, fs.nv ? "night vision" : "colour",
             lens ? "the lens's own picture" : (onImage ? "lens on the eye picture" : "lens off it: static"), v.x, v.y, v.z);
    return true;
}

// ---- [THEATER] the menu / loading presentation, rebuilt for VR (Test34) --------------------------
// Before: a head-locked flat panel on a muted-blue card, glued to your face. Now: a world-locked,
// gently curved screen in a dark room, with the credit plaque under it.
//  - ANCHOR: built in front of where you face when it appears, upright (yaw only). It stays put
//    in the room while you look around. Look away more than ~40 degrees for over a second and it
//    glides back in front of you, so it can never be lost. A recenter re-anchors it at once.
//  - ROOM: a projection layer under everything, drawn per eye from the runtime's own eye poses,
//    so it is true stereo and head-tracked: a near-black void, a floor lit by the screen, and a
//    soft halo round the screen's edge. Cheap enough to redraw every compositor frame.
//  - SCREEN: a cylinder layer when the runtime offers XR_KHR_composition_layer_cylinder, else a
//    flat quad. The image keeps its true aspect; height is capped inside the headset's view.
//  - [Cinema] HeadLocked=1, Environment=0 and Curved=0 each fall back to the old presentation.
static const UINT kEnvW = 960, kEnvH = 1024;
static XrSwapchain g_envSc[2] = { 0, 0 };
static std::vector<ID3D11Texture2D*> g_envImgs[2];
static ID3D11Texture2D* g_envTmp = nullptr;
static ID3D11RenderTargetView* g_envRtv = nullptr;
static ID3D11VertexShader* g_envVS = nullptr;
static ID3D11PixelShader* g_envPS = nullptr;
static ID3D11Buffer* g_envCB = nullptr;
static bool g_envTried = false, g_envReady = false;
static XrVector3f g_thHead = { 0, 0, 0 };   // the viewpoint the theater is built around (LOCAL)
static float g_thYaw = 0.0f;                 // facing, radians, same convention as the screen anchor
static DWORD g_thLastMs = 0, g_thAwaySince = 0;
static bool  g_thFollowing = false;

static const char* kEnvHlsl =
"cbuffer EnvCB : register(b0) { float4 tanB; float4 eyeQ; float4 eyeP; float4 scrC; float4 scrB; };\n"
"struct VOut { float4 p : SV_Position; float2 uv : TEXCOORD0; };\n"
"VOut vs(uint id : SV_VertexID) {\n"
"  float2 t = float2((id << 1) & 2, id & 2);\n"
"  VOut o; o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = t; return o; }\n"
"float3 QRot(float4 q, float3 v) { float3 u = q.xyz; return v + 2.0 * cross(u, cross(u, v) + q.w * v); }\n"
"float4 ps(VOut i) : SV_Target {\n"
"  float3 d = normalize(QRot(eyeQ, float3(lerp(tanB.x, tanB.y, i.uv.x), lerp(tanB.z, tanB.w, i.uv.y), -1.0)));\n"
"  float3 col = lerp(float3(0.0035, 0.0040, 0.0055), float3(0.014, 0.016, 0.021), exp(-abs(d.y) * 5.0));\n"
"  float3 rgt = float3(scrB.x, 0.0, scrB.y);\n"
"  float3 nrm = float3(-scrB.y, 0.0, scrB.x);\n"
"  float den = dot(d, nrm);\n"
"  if (den < -0.001) {\n"
"    float t = dot(scrC.xyz - eyeP.xyz, nrm) / den;\n"
"    if (t > 0.0) {\n"
"      float3 q = eyeP.xyz + d * t - scrC.xyz;\n"
"      float dx = max(abs(dot(q, rgt)) - scrC.w, 0.0);\n"
"      float dy = max(abs(q.y) - scrB.z, 0.0);\n"
"      float r2 = dx * dx + dy * dy;\n"
"      col += float3(0.30, 0.34, 0.42) * (exp(-r2 / 0.08) * 0.10 + exp(-r2 / 1.5) * 0.035);\n"
"    }\n"
"  }\n"
"  if (d.y < -0.0005) {\n"
"    float t = (scrB.w - eyeP.y) / d.y;\n"
"    float3 p = eyeP.xyz + d * t;\n"
"    float2 rel = p.xz - (scrC.xz + nrm.xz * 1.2);\n"
"    float pool = exp(-dot(rel, rel) / 5.0);\n"
"    float2 g = abs(frac(p.xz * 0.75) - 0.5);\n"
"    float lw = 0.012 + 0.004 * t;\n"
"    float grid = smoothstep(0.5 - lw, 0.5, max(g.x, g.y)) * exp(-t * 0.30);\n"
"    float3 fl = float3(0.006, 0.007, 0.009) + float3(0.050, 0.058, 0.072) * pool + grid * 0.010;\n"
"    col = lerp(col, fl, saturate(exp(-t * 0.06)));\n"
"  }\n"
"  float n = frac(sin(dot(i.p.xy, float2(12.9898, 78.233))) * 43758.5453);\n"
"  col += (n - 0.5) * 0.0012;\n"
"  return float4(max(col, 0.0) * eyeP.w, 1.0); }\n";

static float HeadYawLocal() {
    const XrVector3f fwd = QRotate(g_lastLocalQ, { 0.0f, 0.0f, -1.0f });
    return atan2f(-fwd.x, -fwd.z);   // the screen anchor's convention (see the cinema ON branch)
}
static XrQuaternionf TheaterQ() { return { 0.0f, sinf(g_thYaw * 0.5f), 0.0f, cosf(g_thYaw * 0.5f) }; }

// Once per frame while the theater is up. Anchors on first show (or after 1.5 s away), on
// recenter, and glides back in front of you when you have looked well away for a while.
static void TheaterTick() {
    const DWORD now = GetTickCount();
    const float dt = g_thLastMs ? (float)(now - g_thLastMs) * 0.001f : 0.0f;
    if (!g_thValid || (now - g_thLastMs) > 1500u) {
        g_thYaw = HeadYawLocal(); g_thHead = g_lastLocalP; g_thValid = 1;
        g_thFollowing = false; g_thAwaySince = 0;
        XLOG("[OLVR][THEATER] anchored %.2fm ahead, yaw %.1f deg", g_cinemaDistanceM, g_thYaw * 57.2957795f);
    }
    g_thLastMs = now;
    const float diff = WrapPi(HeadYawLocal() - g_thYaw);
    if (!g_thFollowing) {
        if (fabsf(diff) > 0.70f) {               // ~40 degrees off the screen
            if (!g_thAwaySince) g_thAwaySince = now;
            else if (now - g_thAwaySince > 1200u) g_thFollowing = true;
        } else g_thAwaySince = 0;
    }
    if (g_thFollowing) {
        const float k = (dt > 0.1f ? 0.1f : dt) * 3.0f;
        g_thYaw = WrapPi(g_thYaw + diff * k);
        g_thHead.x += (g_lastLocalP.x - g_thHead.x) * k;
        g_thHead.y += (g_lastLocalP.y - g_thHead.y) * k;
        g_thHead.z += (g_lastLocalP.z - g_thHead.z) * k;
        if (fabsf(diff) < 0.03f) { g_thFollowing = false; g_thAwaySince = 0; }
    }
}

static bool EnsureEnvironment() {
    if (!g_thEnv) return false;
    if (g_envReady) return true;
    if (g_envTried) return false;
    g_envTried = true;
    if (!EnsureGammaPipeline()) return false;   // shares its raster/depth state objects
    HMODULE dc = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3DCompile compile = dc ? (PFN_D3DCompile)GetProcAddress(dc, "D3DCompile") : nullptr;
    if (!compile) return false;
    ID3DBlob* vsb = nullptr; ID3DBlob* psb = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(compile(kEnvHlsl, strlen(kEnvHlsl), nullptr, nullptr, nullptr, "vs", "vs_4_0", 0, 0, &vsb, &err)) ||
        FAILED(compile(kEnvHlsl, strlen(kEnvHlsl), nullptr, nullptr, nullptr, "ps", "ps_4_0", 0, 0, &psb, &err))) {
        XLOG("[OLVR][THEATER] room shader compile FAILED%s%s - plain backdrop", err ? ": " : "",
             err ? (const char*)err->GetBufferPointer() : "");
        if (vsb) vsb->Release(); if (psb) psb->Release(); if (err) err->Release();
        return false;
    }
    bool ok = SUCCEEDED(g_d3d11->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_envVS)) &&
              SUCCEEDED(g_d3d11->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_envPS));
    vsb->Release(); psb->Release();
    if (!ok) return false;
    D3D11_BUFFER_DESC bd = {}; bd.ByteWidth = 80; bd.Usage = D3D11_USAGE_DEFAULT; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(g_d3d11->CreateBuffer(&bd, nullptr, &g_envCB))) return false;
    D3D11_TEXTURE2D_DESC td = {}; td.Width = kEnvW; td.Height = kEnvH; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(g_d3d11->CreateTexture2D(&td, nullptr, &g_envTmp)) ||
        FAILED(g_d3d11->CreateRenderTargetView(g_envTmp, nullptr, &g_envRtv))) return false;
    for (int e = 0; e < 2; ++e) {
        XrSwapchainCreateInfo ci = {}; ci.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_VALUE;
        ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT_VALUE | XR_SWAPCHAIN_USAGE_SAMPLED_BIT_VALUE;
        ci.format = (int64_t)DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        ci.sampleCount = 1; ci.width = kEnvW; ci.height = kEnvH;
        ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
        if (!XrSucceeded(g_fn.createSwapchain(g_session, &ci, &g_envSc[e]))) {
            g_envSc[e] = 0; XLOG("[OLVR][THEATER] room swapchain create FAILED - plain backdrop"); return false;
        }
        uint32_t n = 0; g_fn.enumerateSwapchainImages(g_envSc[e], 0, &n, nullptr);
        std::vector<XrSwapchainImageD3D11KHR> imgs(n);
        for (auto& im : imgs) { im.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR_VALUE; im.next = nullptr; im.texture = nullptr; }
        if (!XrSucceeded(g_fn.enumerateSwapchainImages(g_envSc[e], n, &n, (XrSwapchainImageBaseHeader*)imgs.data())))
            return false;
        for (auto& im : imgs) g_envImgs[e].push_back(im.texture);
    }
    g_envReady = true;
    XLOG("[OLVR][THEATER] room ready (%ux%u per eye)", kEnvW, kEnvH);
    return true;
}

struct TheaterLayers {
    XrCompositionLayerProjection env;
    XrCompositionLayerProjectionView envPv[2];
    XrCompositionLayerQuad plaque;
    XrCompositionLayerQuad quad[2];
    XrCompositionLayerCylinderKHR cyl[2];
};

static bool RenderEnvironment(TheaterLayers& t, XrVector3f center, XrVector3f right, float hw, float hh) {
    if (!g_thEnv || !g_rtEyeFovValid || !EnsureEnvironment()) return false;
    for (int e = 0; e < 2; ++e) {
        uint32_t idx = 0;
        XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
        if (!XrSucceeded(g_fn.acquireSwapchainImage(g_envSc[e], &ai, &idx))) return false;
        XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
        g_fn.waitSwapchainImage(g_envSc[e], &wi);
        const XrPosef& ep = g_rtEyePose[e];
        const XrFovf& fv = g_rtEyeFov[e];
        const float cb[20] = {
            tanf(fv.angleLeft), tanf(fv.angleRight), tanf(fv.angleUp), tanf(fv.angleDown),
            ep.orientation.x, ep.orientation.y, ep.orientation.z, ep.orientation.w,
            ep.position.x, ep.position.y, ep.position.z, g_presentFade,
            center.x, center.y, center.z, hw,
            right.x, right.z, hh, g_thHead.y - 1.30f };
        g_ctx->UpdateSubresource(g_envCB, 0, nullptr, cb, 0, 0);
        D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)kEnvW, (float)kEnvH, 0.0f, 1.0f };
        const float blendFactor[4] = { 0, 0, 0, 0 };
        g_ctx->IASetInputLayout(nullptr);
        g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_ctx->VSSetShader(g_envVS, nullptr, 0);
        g_ctx->PSSetShader(g_envPS, nullptr, 0);
        g_ctx->PSSetConstantBuffers(0, 1, &g_envCB);
        g_ctx->OMSetBlendState(nullptr, blendFactor, 0xffffffff);
        g_ctx->OMSetDepthStencilState(g_blitDepth, 0);
        g_ctx->OMSetRenderTargets(1, &g_envRtv, nullptr);
        g_ctx->RSSetState(g_blitRaster);
        g_ctx->RSSetViewports(1, &vp);
        g_ctx->Draw(3, 0);
        ID3D11RenderTargetView* nulrt = nullptr;
        g_ctx->OMSetRenderTargets(1, &nulrt, nullptr);
        if (idx < g_envImgs[e].size()) g_ctx->CopyResource(g_envImgs[e][idx], g_envTmp);
        XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
        g_fn.releaseSwapchainImage(g_envSc[e], &ri);
        t.envPv[e] = {};
        t.envPv[e].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW_VALUE;
        t.envPv[e].pose = ep; t.envPv[e].fov = fv;
        t.envPv[e].subImage.swapchain = g_envSc[e];
        t.envPv[e].subImage.imageArrayIndex = 0;
        t.envPv[e].subImage.imageRect.offset = { 0, 0 };
        t.envPv[e].subImage.imageRect.extent = { (int32_t)kEnvW, (int32_t)kEnvH };
    }
    t.env = {};
    t.env.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VALUE;
    t.env.layerFlags = 0; t.env.space = g_localSpace; t.env.viewCount = 2; t.env.views = t.envPv;
    return true;
}

// Appends room, plaque and screen to layers[] and returns the new count. stereo: sc[0]/sc[1] are
// per-eye images (LEFT/RIGHT visibility); otherwise sc[0] is shown to both eyes. aspect is the
// image's TRUE width/height. *sceneLayer is set when a projection layer (the room) went out.
static int AddTheaterLayers(TheaterLayers& t, const XrCompositionLayerBaseHeader** layers, int n,
                            bool stereo, const XrSwapchain sc[2], XrRect2Di rect, float aspect,
                            bool* sceneLayer) {
    TheaterTick();
    const float D = g_cinemaDistanceM;
    const float up = -0.06f;   // centre a touch below eye level
    const float hfy = (g_rtFovY > 0.2f) ? g_rtFovY : 1.75f;
    if (!(aspect > 0.2f && aspect < 5.0f)) aspect = 16.0f / 9.0f;
    float A = g_cinemaWidthDeg * 0.0174532925f;
    const float maxV = hfy * 0.90f;   // keep the whole picture inside the headset's vertical view
    if (2.0f * atanf(tanf(A * 0.5f) / aspect) > maxV) A = 2.0f * atanf(tanf(maxV * 0.5f) * aspect);
    const bool curved = g_cylinderExt && g_thCurved;
    const float qw = 2.0f * D * tanf(A * 0.5f), qh = qw / aspect;
    const float halfH = curved ? (D * A / aspect) * 0.5f : qh * 0.5f;
    const XrQuaternionf yq = TheaterQ();
    const XrVector3f right = QRotate(yq, { 1.0f, 0.0f, 0.0f });
    const XrVector3f fwdOff = QRotate(yq, { 0.0f, up, -D });
    const XrVector3f center = { g_thHead.x + fwdOff.x, g_thHead.y + fwdOff.y, g_thHead.z + fwdOff.z };

    if (RenderEnvironment(t, center, right, qw * 0.5f, halfH)) {
        layers[n++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&t.env);
        *sceneLayer = true;
    }
    if (EnsureBrandScene()) {
        // The credit plaque: the backdrop's own footer strip (logo + "VR mod made by StrikeZoneVR").
        const XrVector3f po = QRotate(yq, { 0.0f, up - halfH - 0.14f, -D + 0.02f });
        t.plaque = {};
        t.plaque.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
        t.plaque.layerFlags = 0; t.plaque.space = g_localSpace;
        t.plaque.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
        t.plaque.subImage.swapchain = g_brandSc; t.plaque.subImage.imageArrayIndex = 0;
        t.plaque.subImage.imageRect.offset = { 260, 830 };
        t.plaque.subImage.imageRect.extent = { 1200, kBrandH - 830 };
        t.plaque.pose.orientation = yq;
        t.plaque.pose.position = { g_thHead.x + po.x, g_thHead.y + po.y, g_thHead.z + po.z };
        t.plaque.size.width = 0.80f;
        t.plaque.size.height = 0.80f * (float)(kBrandH - 830) / 1200.0f;
        layers[n++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&t.plaque);
    }
    const int count = stereo ? 2 : 1;
    for (int e = 0; e < count; ++e) {
        const XrEyeVisibility vis = !stereo ? XR_EYE_VISIBILITY_BOTH_VALUE
                                  : (e == 0 ? XR_EYE_VISIBILITY_LEFT_VALUE : XR_EYE_VISIBILITY_RIGHT_VALUE);
        if (curved) {
            XrCompositionLayerCylinderKHR& c = t.cyl[e];
            c = {};
            c.type = XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR_VALUE;
            c.layerFlags = 0; c.space = g_localSpace; c.eyeVisibility = vis;
            c.subImage.swapchain = sc[e]; c.subImage.imageRect = rect; c.subImage.imageArrayIndex = 0;
            c.pose.orientation = yq;
            c.pose.position = { g_thHead.x, g_thHead.y + up, g_thHead.z };   // the cylinder's axis
            c.radius = D; c.centralAngle = A; c.aspectRatio = aspect;
            layers[n++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&c);
        } else {
            XrCompositionLayerQuad& q = t.quad[e];
            q = {};
            q.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
            q.layerFlags = 0; q.space = g_localSpace; q.eyeVisibility = vis;
            q.subImage.swapchain = sc[e]; q.subImage.imageRect = rect; q.subImage.imageArrayIndex = 0;
            q.pose.orientation = yq; q.pose.position = center;
            q.size.width = qw; q.size.height = qh;
            layers[n++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q);
        }
    }
    return n;
}

static bool EnsureEyeHold() {
    for (int e = 0; e < 2; ++e) {
        if (g_eyeHold[e]) continue;
        D3D11_TEXTURE2D_DESC td = {}; td.Width = g_srcEyeW; td.Height = g_srcEyeH;
        td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(g_d3d11->CreateTexture2D(&td, nullptr, &g_eyeHold[e]))) { g_eyeHold[e] = nullptr; return false; }
    }
    return true;
}

static bool BlitHoldToEye(XrSwapchain sc, std::vector<ID3D11Texture2D*>& imgs, ID3D11Texture2D* src) {
    if (!sc || imgs.empty() || !src) return false;
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (!XrSucceeded(g_fn.acquireSwapchainImage(sc, &ai, &idx))) return false;
    bool ok = false;
    XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
    if (XrSucceeded(g_fn.waitSwapchainImage(sc, &wi))) {
        if (ShaderBlitReady())
            ok = GammaDraw(src, g_srcEyeW, g_srcEyeH, 0.0f, g_srcEyeW, g_srcEyeH,
                           0.0f, (float)g_srcEyeW, imgs[idx], g_eyeW, g_eyeH);   // opaque alpha
        else if (g_eyeW == g_srcEyeW && g_eyeH == g_srcEyeH) {
            g_ctx->CopyResource(imgs[idx], src); ok = true;
        }
    }
    XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
    g_fn.releaseSwapchainImage(sc, &ri);
    return ok;
}

static bool BlitToSwapchain() {
    if (!g_swapchain || g_xrImages.empty()) return false;
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai = {}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO_VALUE;
    if (!XrSucceeded(g_fn.acquireSwapchainImage(g_swapchain, &ai, &idx))) return false;
    bool copied = false;
    XrSwapchainImageWaitInfo wi = {}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO_VALUE; wi.timeout = XR_INFINITE_DURATION_VALUE;
    if (XrSucceeded(g_fn.waitSwapchainImage(g_swapchain, &wi))) {
        if (ShaderBlitReady()) {   // opaque alpha, see BlitHalfToEye
            copied = GammaDraw(g_uploadTex, g_w, g_h, 0.0f, g_w, g_h, 0.0f, (float)g_w, g_xrImages[idx], g_w, g_h);
        } else {
            g_ctx->CopyResource(g_xrImages[idx], g_uploadTex);   // UNORM -> UNORM_SRGB, same typeless family
            copied = true;
        }
    }
    XrSwapchainImageReleaseInfo ri = {}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO_VALUE;
    g_fn.releaseSwapchainImage(g_swapchain, &ri);
    return copied;
}

// [PERF] Present-to-Present interval and the 5 s report. Runs whether or not a session is up, so
// a flat run measures the game alone.
// (Test46) [HITCH] when each long frame happened (seconds on the log's clock) - a pattern ("every 10 s")
// shows up next to whatever else the log says happened then.
extern "C" double OLProxy_LogTime();
static double g_hitchAt[12]; static float g_hitchMs[12]; static int g_hitchN = 0;
static void PerfFrame() {
    static double s_last = 0.0, s_report = 0.0;
    const double now = PerfNowMs();
    if (s_last > 0.0) {
        const double dt = now - s_last;
        g_pfFrame.Add(dt);
        if (dt > 25.0) ++g_pfHitch25;
        if (dt > 50.0) ++g_pfHitch50;
        if (dt > 40.0 && g_hitchN < 12) { g_hitchAt[g_hitchN] = OLProxy_LogTime(); g_hitchMs[g_hitchN] = (float)dt; ++g_hitchN; }
    }
    s_last = now;
    if (s_report <= 0.0) s_report = now;
    if (now - s_report < 5000.0) return;
    PerfStat up, wa, xb, xe, xs, xl; long xf, xn;
    AcquireSRWLockExclusive(&g_pfLock);
    up = g_pfUpload; wa = g_pfWait; xf = g_pfXrFrames; xn = g_pfXrNew;
    xb = g_pfBegin; xe = g_pfEnd; xs = g_pfStep; xl = g_pfLoop;
    g_pfUpload = PerfStat(); g_pfWait = PerfStat(); g_pfXrFrames = 0; g_pfXrNew = 0;
    g_pfBegin = PerfStat(); g_pfEnd = PerfStat(); g_pfStep = PerfStat(); g_pfLoop = PerfStat();
    ReleaseSRWLockExclusive(&g_pfLock);
    const double secs = (now - s_report) / 1000.0;
    XLOG("[OLVR][PERF] game %.1f fps (frame avg %.1f max %.1f ms, >25ms %ld, >50ms %ld) | present avg %.2f max %.2f | "
         "grab avg %.2f max %.2f | copy avg %.2f max %.2f | pace avg %.2f max %.2f | xr %.1f fps, new %.1f/s, "
         "wait avg %.2f max %.2f, upload avg %.2f max %.2f | period %.2f ms%s",
         g_pfFrame.n / secs, g_pfFrame.Avg(), g_pfFrame.mx, g_pfHitch25, g_pfHitch50,
         g_pfPresent.Avg(), g_pfPresent.mx,
         g_pfGrab.Avg(), g_pfGrab.mx, g_pfCopy.Avg(), g_pfCopy.mx, g_pfPace.Avg(), g_pfPace.mx,
         xf / secs, xn / secs, wa.Avg(), wa.mx, up.Avg(), up.mx, g_displayPeriodUs / 1000.0,
         g_devPublish ? (g_devUpload ? "" : " [bridge upload SKIPPED by dev]") : " [publishing PAUSED by dev]");
    XLOG("[OLVR][PERF] path=%s%s, eye redraws skipped for repeated frames: %.1f/s",
         g_exState > 0 ? "D3D9Ex shared frames (GPU hand-over)" :
         (g_zcState > 0 ? "zero-copy (GPU hand-over)" : (g_asyncReadback ? "async readback" : "readback")),
         (g_zcSkipped || g_exSkipped) ? " - some frames not sent, all shared frames busy" : "", InterlockedExchange(&g_heldEyeFrames, 0) / secs);
    g_zcSkipped = 0; g_exSkipped = 0;
    OLProxy_ExStatReport();   // (Test47) [EXSTAT] the game's resource creates / locks this period
    XLOG("[OLVR][XRTIMING] loop avg %.2f max %.2f | begin avg %.2f max %.2f | end avg %.2f max %.2f | "
         "display step avg %.2f max %.2f ms",
         xl.Avg(), xl.mx, xb.Avg(), xb.mx, xe.Avg(), xe.mx, xs.Avg(), xs.mx);
    {
        // [CPUSPLIT] (Test36) how busy the game thread (builds the views) and the render thread
        // (presents) were over the window: near 100% marks the thread that limits the frame rate.
        static ULONGLONG s_prev[2] = { 0, 0 };
        const DWORD tids[2] = { g_tidGame, g_tidPresent };
        int pct[2] = { -1, -1 };
        for (int i = 0; i < 2; ++i) {
            if (!tids[i]) continue;
            HANDLE th = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tids[i]);
            if (!th) continue;
            FILETIME c, e, k, u;
            if (GetThreadTimes(th, &c, &e, &k, &u)) {
                const ULONGLONG t = ((ULONGLONG)k.dwHighDateTime << 32 | k.dwLowDateTime) +
                                    ((ULONGLONG)u.dwHighDateTime << 32 | u.dwLowDateTime);
                if (s_prev[i] && t >= s_prev[i]) pct[i] = (int)((double)(t - s_prev[i]) / 1e4 / (secs * 1000.0) * 100.0 + 0.5);
                s_prev[i] = t;
            }
            CloseHandle(th);
        }
        XLOG("[OLVR][CPUSPLIT] game thread %d%% busy, render thread %d%% busy", pct[0], pct[1]);
    }
    {
        // [FRAMETAG] tagged = Presents whose pose came from the frame the renderer identified;
        // oldRuleWrong = of those, how many the Test35 rule (live pose at Present) got wrong.
        const long t0 = InterlockedExchange(&g_tagStat[0], 0), t1 = InterlockedExchange(&g_tagStat[1], 0);
        const long t2 = InterlockedExchange(&g_tagStat[2], 0), t3 = InterlockedExchange(&g_tagStat[3], 0);
        const long d = InterlockedExchange(&g_tagDiff, 0);
        XLOG("[OLVR][FRAMETAG] tagged %ld untagged %ld (view lookups hit %ld miss %ld) oldRuleWrong %ld "
             "| threads: views=%lu present=%lu%s",
             t0, t1, t3, t2, d, (unsigned long)g_tidGame, (unsigned long)g_tidPresent,
             !g_latchOn ? " [latch OFF]" : "");
    }

    if (g_hitchN) {
        char hl[400]; int at = sprintf_s(hl, "[OLVR][HITCH] frames over 40 ms:");
        for (int i = 0; i < g_hitchN && at < 360; ++i) at += sprintf_s(hl + at, sizeof(hl) - at, " %.2fs %.0fms", g_hitchAt[i], g_hitchMs[i]);
        XLOG("%s", hl);
        g_hitchN = 0;
    }
    g_pfFrame = PerfStat(); g_pfGrab = PerfStat(); g_pfCopy = PerfStat(); g_pfPace = PerfStat();
    g_pfPresent = PerfStat();
    g_pfHitch25 = g_pfHitch50 = 0;
    s_report = now;
}

// [PACING] see g_paceToHeadset. Waits only while the previous frame is still unconsumed.
static void PaceToHeadset() {
    const LONG periodUs = g_displayPeriodUs;
    if (!g_paceToHeadset || !g_running || periodUs <= 0) return;
    const double t0 = PerfNowMs(), limit = t0 + periodUs / 1000.0 * 1.25;
    for (;;) {
        AcquireSRWLockShared(&g_pubLock);
        const bool taken = g_pubSeq == 0 || g_conSeq >= g_pubSeq;
        ReleaseSRWLockShared(&g_pubLock);
        if (taken) break;
        const double now = PerfNowMs();
        if (now >= limit) break;
        if (limit - now > 1.5) Sleep(1); else YieldProcessor();
    }
    g_pfPace.Add(PerfNowMs() - t0);
}

void OLXR_OnPresent(IDirect3DDevice9* dev) {
    PerfFrame();
    if (g_dead || !dev) return;
    g_lastPresentMs = NowMs();   // [EXITGUARD] heartbeat the worker watches
    if (!g_inited) {
        IDirect3DSurface9* bb = nullptr;
        if (FAILED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) || !bb) return;
        D3DSURFACE_DESC sd; bb->GetDesc(&sd); bb->Release();
        if (sd.Width < 64 || sd.Height < 64) return;    // not a real scene yet
        if (!Init(sd.Width, sd.Height)) return;
    }

    // [SUBMITTHREAD] With the worker on, the game thread's whole job is publishing frames; the
    // worker owns events, pacing and every XR call. Publishing is gated on the session running
    // so a headset on standby does not pay the readback for nothing.
    if (g_useWorker) {
        PresentPopTag();   // [FRAMETAG] every Present, published or not
        if (g_running && g_devPublish) { PaceToHeadset(); PublishBackbuffer(dev); }
        OLUE3_FrameTick();   // once per GAME frame - see the note at the old call site
        return;
    }

    PollEvents();
    if (!g_running) return;

    // Follow the live backbuffer size. Cheap, and it is the difference between a picture and
    // a black headset the moment the game changes resolution.
    {
        IDirect3DSurface9* bb = nullptr;
        if (SUCCEEDED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) && bb) {
            D3DSURFACE_DESC sd;
            if (SUCCEEDED(bb->GetDesc(&sd)) && sd.Width >= 64 && sd.Height >= 64 &&
                (sd.Width != g_w || sd.Height != g_h)) {
                bb->Release();
                if (!RebuildForSize(sd.Width, sd.Height)) return;
            } else bb->Release();
        }
    }

    // [PACING] Capture the backbuffer BEFORE xrWaitFrame. GetRenderTargetData drains the whole
    // GPU pipeline behind the just-rendered 4K frame, and the lock+upload moves ~33MB on the CPU.
    // Doing that between waitFrame and endFrame put it inside the compositor's armed deadline;
    // SteamVR booked the overruns as frame timeouts (903 episodes in one 30s session) and held
    // the game cross-faded into its backdrop - the "transparent over the environment" look. The
    // work is identical out here; only the deadline exposure changes. Mode 2 (SFR) re-captures
    // inside the window on purpose: its left-eye upload overwrites g_uploadTex, so the right eye
    // must take the backbuffer again after it.
    const bool preCaptured = CaptureBackbuffer(dev);
    RunXrFrame(dev, preCaptured);
    OLUE3_FrameTick();   // once per Present, exactly as before the worker existed
}

// One full compositor frame: waitFrame -> pose -> layers -> endFrame. Called from the game's
// Present in inline mode (dev = the D3D9 device) or from the worker's loop (dev = null; the
// image is whatever ConsumeUpload last put in g_uploadTex, and the pose tag is the one that
// was published WITH that image). The SFR/AER lanes need the live device and quietly stand
// down under the worker - both are dev-only research lanes, mode split is what ships.
static bool RunXrFrame(IDirect3DDevice9* dev, bool preCaptured) {
    XrFrameWaitInfo fwi = {}; fwi.type = XR_TYPE_FRAME_WAIT_INFO_VALUE;
    XrFrameState fs = {}; fs.type = XR_TYPE_FRAME_STATE_VALUE;
    const double tw = PerfNowMs();
    if (!XrSucceeded(g_fn.waitFrame(g_session, &fwi, &fs))) return false;
    {
        const double dt = PerfNowMs() - tw;
        AcquireSRWLockExclusive(&g_pfLock); g_pfWait.Add(dt); ++g_pfXrFrames; ReleaseSRWLockExclusive(&g_pfLock);
        if (fs.predictedDisplayPeriod > 0)
            InterlockedExchange(&g_displayPeriodUs, (LONG)(fs.predictedDisplayPeriod / 1000));
        // [XRTIMING] loop period and the display-time step
        static double s_lastLoop = 0.0; static XrTime s_lastDisp = 0;
        AcquireSRWLockExclusive(&g_pfLock);
        if (s_lastLoop > 0.0) g_pfLoop.Add(tw - s_lastLoop);
        if (s_lastDisp && fs.predictedDisplayTime > s_lastDisp && fs.predictedDisplayTime - s_lastDisp < 1000000000LL)
            g_pfStep.Add((double)(fs.predictedDisplayTime - s_lastDisp) / 1e6);
        ReleaseSRWLockExclusive(&g_pfLock);
        s_lastLoop = tw; s_lastDisp = fs.predictedDisplayTime;
    }

    XrFrameBeginInfo fbi = {}; fbi.type = XR_TYPE_FRAME_BEGIN_INFO_VALUE;
    {
        const double tb = PerfNowMs();
        g_fn.beginFrame(g_session, &fbi);
        const double db = PerfNowMs() - tb;
        AcquireSRWLockExclusive(&g_pfLock); g_pfBegin.Add(db); ReleaseSRWLockExclusive(&g_pfLock);
    }

    // [XRINPUT] Controllers, on the worker, once per frame, at the same predicted display time
    // the head pose uses - so hands and head are sampled for the same instant. This is the ONLY
    // place actions are synced; the game thread just reads the published snapshot.
    SyncInput(fs.predictedDisplayTime);

    // POSE (2026-07-23), both earlier shapes falsified LIVE, keep this history:
    //  - Latch the render pose, predict for the CURRENT display: declared sits a full frame
    //    behind the display pose, the compositor timewarps a frame of yaw per turn - reads as
    //    the view warping on every head turn (attempt 1).
    //  - Declare the FRESH pose over a frame-old image (the literal ME1 read): every pacing
    //    hiccup between game fps and headset refresh shows raw - reads as noticeably shaky
    //    head tracking (attempt 2).
    // The actual fix is PREDICTION, not freshness: the locate below asks for the head pose one
    // display period AHEAD - the time the frame the camera is about to render will actually hit
    // the screen. The camera renders with it; THIS latch carries it to the submit one Present
    // later, when that frame is displayed. Declared == rendered again (no shake), and the
    // declared pose targets the right instant (no systematic timewarp, no warp on turns). What
    // remains for the compositor is prediction error, which is what timewarp is FOR.
    // [SUBMITTHREAD] Under the worker the image being submitted is the last PUBLISHED game
    // frame, so its tag is the pose that travelled with it - not the latest locate, which may
    // be several compositor frames newer than the pixels.
    const XrPosef poseForThisImage = dev ? g_headPoseApp : g_conPose;
    const long    poseValidForThisImage = dev ? g_olPoseValid : g_conPoseValid;

    // ---- head pose: software baseline from LOCAL, current-minus-baseline thereafter ----
    if (g_recenterReq) {
        XrViewLocateInfo li = {}; li.type = XR_TYPE_VIEW_LOCATE_INFO_VALUE;
        li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_VALUE;
        li.displayTime = fs.predictedDisplayTime; li.space = g_localSpace;
        XrViewState ls = {}; ls.type = XR_TYPE_VIEW_STATE_VALUE;
        XrView lv[2] = {}; lv[0].type = XR_TYPE_VIEW_VALUE; lv[1].type = XR_TYPE_VIEW_VALUE; uint32_t ln = 0;
        if (XrSucceeded(g_fn.locateViews(g_session, &li, &ls, 2, &ln, lv)) &&
            (ls.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT_VALUE)) {
            // YAW ONLY. Capturing the full orientation makes "forward" mean whatever tilt your
            // head happened to have at that instant, and every later pose is measured against
            // that tilted frame - so the horizon is permanently off and the world reads as
            // "not straight". It bites hardest on the automatic baseline at startup, which
            // fires while the headset is still being put on. Pitch and roll must stay absolute
            // so the world stays gravity-aligned; only the direction you face is adopted.
            // Extraction is the round trip already proven in this file (see the cinema-screen
            // anchor): rotating (0,0,-1) by +theta lands at (-sin,0,-cos), so
            // atan2(-fwd.x,-fwd.z) recovers +theta and the quaternion takes +yaw/2.
            const XrQuaternionf ql = QNormalize(lv[0].pose.orientation);
            const XrVector3f fwd = QRotate(ql, { 0.0f, 0.0f, -1.0f });
            const float horiz = fwd.x * fwd.x + fwd.z * fwd.z;
            if (horiz > 1e-6f) {
                const float yaw = atan2f(-fwd.x, -fwd.z);
                g_baseQ = { 0.0f, sinf(yaw * 0.5f), 0.0f, cosf(yaw * 0.5f) };
            }   // straight up or down: yaw is meaningless there, keep the previous facing
            g_baseP = PoseCenter(lv);
            g_recenterReq = 0; g_centered = 1; g_olPoseValid = 0;
            InterlockedIncrement(&g_recenterGeneration);
            // [HEADPOS] A player-requested recenter is a trusted position zero. The boot one is
            // not: it starts the pickup-then-still watch instead, and position stays inert
            // until that (or a G press) arms it.
            const bool manual = g_recenterManual != 0;
            g_recenterManual = 0;
            if (manual) {
                g_posArmed = 1; g_posAutoDone = 1;
            } else {
                g_posArmed = 0; g_posPickedUp = 0; g_settleSince = 0;
            }
            XLOG("[OLVR][XR] recenter: baseline captured (yaw only - the horizon stays level), "
                 "position %s", manual ? "armed (player recenter)"
                                       : "waiting - arms on the first still pose after the headset is picked up, or on recenter");
        }
    } else if (g_centered) {
        {   // (Beta 1.1) [BODYTURN] the yaw Miles just turned, taken out of the head's relative yaw (see above)
            const LONG pm = InterlockedExchange(&g_bodyTurnPendMilli, 0);
            if (pm) {
                const float d = (float)pm * 0.001f * 0.0174532925f;                         // right-positive
                const XrQuaternionf ry = { 0.0f, sinf(-d * 0.5f), 0.0f, cosf(-d * 0.5f) };  // XR yaw is left-positive
                g_baseQ = QNormalize(QMul(ry, g_baseQ));
                // relative = conj(base) * raw, so every relative orientation turns LEFT by d: the filters'
                // state too, or they would read the turn as head motion and smooth it in
                const XrQuaternionf rr = { 0.0f, sinf(d * 0.5f), 0.0f, cosf(d * 0.5f) };
                g_smoothQuat = QNormalize(QMul(rr, g_smoothQuat));
                g_handSmoothQ = QNormalize(QMul(rr, g_handSmoothQ));
            }
        }
        XrViewLocateInfo vli = {}; vli.type = XR_TYPE_VIEW_LOCATE_INFO_VALUE;
        vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_VALUE;
        // One period AHEAD: this pose is consumed by the camera for the frame rendered during
        // the NEXT interval, displayed at roughly the next Present. See the pose note above.
        vli.displayTime = fs.predictedDisplayTime + fs.predictedDisplayPeriod;
        vli.space = g_localSpace;
        XrViewState vs = {}; vs.type = XR_TYPE_VIEW_STATE_VALUE;
        XrView av[2] = {}; av[0].type = XR_TYPE_VIEW_VALUE; av[1].type = XR_TYPE_VIEW_VALUE; uint32_t nv = 0;
        if (XrSucceeded(g_fn.locateViews(g_session, &vli, &vs, 2, &nv, av)) &&
            (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT_VALUE)) {
            // Raw LOCAL-space head pose, kept for anchoring the cinema screen in the world.
            g_lastLocalQ = QNormalize(av[0].pose.orientation);
            g_lastLocalP = PoseCenter(av);
            g_lastLocalValid = 1;
            XrQuaternionf qraw = QMul(QConj(g_baseQ), QNormalize(av[0].pose.orientation));
            if (g_devLookYaw != 0.0f || g_devLookPitch != 0.0f) {
                // [DEVBRIDGE] "look": a fixed head turn/tilt on a headset that never moves
                const float hy = g_devLookYaw * 0.00872664626f, hp = g_devLookPitch * 0.00872664626f;
                const XrQuaternionf qy = { 0.0f, sinf(hy), 0.0f, cosf(hy) }, qp = { sinf(hp), 0.0f, 0.0f, cosf(hp) };
                qraw = QMul(QMul(qy, qp), qraw);
            }
            if (g_devHeadWave) {
                // [DEVBRIDGE] "headwave 1": a steady +/-34 deg head turn (~70 deg/s peak) on a headset
                // that never moves, so pose pairing errors become measurable.
                const float a = 0.6f * sinf((float)((double)(vli.displayTime % 3141592654LL) * 2e-9));
                qraw = QMul({ 0.0f, sinf(a * 0.5f), 0.0f, cosf(a * 0.5f) }, qraw);
            }
            XrVector3f c = PoseCenter(av);
            if (g_devHeadOff[0] != 0.0f || g_devHeadOff[1] != 0.0f || g_devHeadOff[2] != 0.0f) {
                const XrVector3f o = QRotate(g_baseQ, { g_devHeadOff[0], g_devHeadOff[1], g_devHeadOff[2] });
                c.x += o.x; c.y += o.y; c.z += o.z;
            }
            // [HEADPOS] Automatic arming, once: pickup (moved > kPickupM from the boot
            // baseline) then still (within kStillM of one spot for kStillNs). Position zero
            // becomes that still spot. Only the position baseline moves - g_baseQ (yaw) is
            // left exactly as the boot recenter took it, so nothing rotates under the player.
            if (!g_posArmed && !g_posAutoDone) {
                const float bx = c.x - g_baseP.x, by = c.y - g_baseP.y, bz = c.z - g_baseP.z;
                if (!g_posPickedUp) {
                    if (bx * bx + by * by + bz * bz > kPickupM * kPickupM) {
                        g_posPickedUp = 1; g_settleRef = c; g_settleSince = vli.displayTime;
                        XLOG("[OLVR][HEADPOS] headset moved off the boot baseline - watching for it to hold still");
                    }
                } else {
                    const float sx = c.x - g_settleRef.x, sy = c.y - g_settleRef.y, sz = c.z - g_settleRef.z;
                    if (sx * sx + sy * sy + sz * sz > kStillM * kStillM) {
                        g_settleRef = c; g_settleSince = vli.displayTime;   // still moving: restart
                    } else if (vli.displayTime - g_settleSince >= kStillNs) {
                        g_baseP = c;
                        g_posArmed = 1; g_posAutoDone = 1;
                        XLOG("[OLVR][HEADPOS] held still - position baseline armed automatically "
                             "(recenter any time to re-take it)");
                    }
                }
            }
            XrVector3f dp = { c.x - g_baseP.x, c.y - g_baseP.y, c.z - g_baseP.z };
            XrVector3f relP = QRotate(QConj(g_baseQ), dp);
            // [BODYFOLLOW] across the floor only; a follow moves the hands' frame identically.
            // [ROOMWALK] (Test44) while Miles may walk with you, each follow is his step too.
            const bool rwOn = GetTickCount64() < g_rwAllowUntil;
            if (g_bodyAnchor && (g_bodyFollow || rwOn) && g_posArmed) {
                static XrVector3f s_prevC = { 0, 0, 0 };
                static XrTime s_prevT = 0, s_stillSince = 0;
                const float dts = s_prevT ? (float)((double)(vli.displayTime - s_prevT) * 1e-9) : 0.0f;
                const float mvx = c.x - s_prevC.x, mvz = c.z - s_prevC.z;
                const float speed = dts > 0.0f ? sqrtf(mvx * mvx + mvz * mvz) / dts : 0.0f;
                s_prevC = c; s_prevT = vli.displayTime;
                if (speed > 0.06f || !s_stillSince) s_stillSince = vli.displayTime;
                const XrVector3f lim = ClampBodyPose(relP);
                XrVector3f shift = { relP.x - lim.x, 0.0f, relP.z - lim.z };          // past the bounds: drag
                const bool still = (vli.displayTime - s_stillSince) > 2500000000LL;       // 2.5 s still
                if (still && dts > 0.0f && dts < 0.1f) {
                    const float k = dts / 9.0f;                                           // ~9 s to settle back under you
                    shift.x += (relP.x - shift.x) * k;
                    shift.z += (relP.z - shift.z) * k;
                }
                if (shift.x != 0.0f || shift.z != 0.0f) {
                    const XrVector3f w = QRotate(g_baseQ, shift);
                    g_baseP.x += w.x; g_baseP.z += w.z;
                    // [ROOMWALK] ...and Miles takes the same step (faster than 3.5 m/s is the tracking, not you)
                    const bool rwStep = rwOn && speed < 3.5f;
                    if (rwStep) { g_rwCum[0] += shift.x; g_rwCum[1] += shift.z; }
                    static float s_moved = 0.0f, s_handed = 0.0f; static XrTime s_logT = 0;
                    const float step = sqrtf(shift.x * shift.x + shift.z * shift.z);
                    s_moved += step; if (rwStep) s_handed += step;
                    if (s_moved > 0.03f && vli.displayTime - s_logT > 3000000000LL) {
                        s_logT = vli.displayTime;
                        XLOG("[OLVR][BODYFOLLOW] the body followed you %.0f cm (%s; %.0f cm of it handed to Miles's feet); head now (%.2f, %.2f) m from it",
                             s_moved * 100.0f, still ? "you settled in a new spot" : "you went past its bounds", s_handed * 100.0f,
                             relP.x - shift.x, relP.z - shift.z);
                        s_moved = 0.0f; s_handed = 0.0f;
                    }
                    dp = { c.x - g_baseP.x, c.y - g_baseP.y, c.z - g_baseP.z };
                    relP = QRotate(QConj(g_baseQ), dp);
                }
            }
            const XrVector3f rawRelP = relP;   // [HANDSYNC] before the hand-cam blend and body clamp
            // SPEED-ADAPTIVE QUATERNION SMOOTHING (ME1's model). Filter ONE quaternion here,
            // upstream of the angle extraction, so both the render angles and the declared tag
            // come from the same smoothed value (a per-axis Euler filter, the old approach,
            // desynced render from tag - ME1's own history documents that bug). The follow factor
            // ramps from heavy-at-rest to 1.0 (raw) once the frame delta passes kSnapDeg, so a
            // real turn has zero added lag while a still head stops jittering.
            // [HEADPOS] Position is NOT filtered - ME1's model exactly (xr_session.cpp: headPos is
            // the raw view centre, only the quaternion is smoothed). Filtering it (tried
            // 2026-08-17, read as worse) lags the lean behind the real head.
            if (!g_smoothInit) { g_smoothQuat = qraw; g_smoothInit = true; }
            if (g_smoothing > 0.001f) {
                const float baseFollow = 1.0f - g_smoothing * 0.95f;
                float t = QuatAngleDeg(g_smoothQuat, qraw) / kSnapDeg;
                if (t > 1.0f) t = 1.0f;
                g_smoothQuat = NlerpQuat(g_smoothQuat, qraw, baseFollow + (1.0f - baseFollow) * t);
            } else {
                g_smoothQuat = qraw;
            }
            XrQuaternionf q = g_smoothQuat;

            // [HANDCAM] HAND THE CAMERA TO THE RIGHT CONTROLLER. Everything downstream of this
            // point - the angle extraction, the declared orientation rebuild, the position
            // publish - is untouched and does not know the difference, which is exactly why
            // declared still equals rendered. Blend so strength 0 is the head, 1 is the hand,
            // and anything between is a real mix rather than a hard switch (a hard switch at the
            // grip press snaps the world; a 4-frame blend does not).
            {
                static float s_blend = 0.0f;
                const bool want = (g_handCamOn && g_handCamActive && g_handRelValid);
                const float target = want ? g_handCamStrength : 0.0f;
                s_blend += (target - s_blend) * 0.25f;          // ~4 frames in and out
                if (s_blend < 0.001f) { s_blend = 0.0f; g_handSmoothInit = false; }
                if (s_blend > 0.0f && g_handRelValid) {
                    // Same speed-adaptive filter as the head, own state: a hand shakes more than
                    // a head, and an unfiltered lens is unwatchable.
                    if (!g_handSmoothInit) { g_handSmoothQ = g_handRelQ; g_handSmoothInit = true; }
                    if (g_smoothing > 0.001f) {
                        const float bf = 1.0f - g_smoothing * 0.95f;
                        float t = QuatAngleDeg(g_handSmoothQ, g_handRelQ) / kSnapDeg;
                        if (t > 1.0f) t = 1.0f;
                        g_handSmoothQ = NlerpQuat(g_handSmoothQ, g_handRelQ, bf + (1.0f - bf) * t);
                    } else {
                        g_handSmoothQ = g_handRelQ;
                    }
                    q = NlerpQuat(g_smoothQuat, g_handSmoothQ, s_blend);
                    // POSITION only moves to the hand when the game will ACTUALLY move the camera
                    // there - i.e. lean is on and armed, because that is the code path that
                    // applies it. Declaring the hand position for an image the game rendered from
                    // the head is a ~40 cm declared/rendered mismatch, and that mismatch is
                    // exactly what produced the swimming this project spent a day chasing.
                    if (OLUE3_GetHeadPos() && g_posArmed) {
                        relP.x += (g_handRelP.x - relP.x) * s_blend;
                        relP.y += (g_handRelP.y - relP.y) * s_blend;
                        relP.z += (g_handRelP.z - relP.z) * s_blend;
                    }
                }
            }

            // Apply the body boundary after every possible pose owner (including the optional
            // hand-camera path) and before publishing. Both UE3 and OpenXR therefore receive the
            // exact same bounded position for this frame.
            relP = ClampBodyPose(relP);

            float fx = -2.0f * (q.x * q.z + q.w * q.y);
            float fy = -2.0f * (q.y * q.z - q.w * q.x);
            float fz = -(1.0f - 2.0f * (q.x * q.x + q.y * q.y));
            float pyc = (fy < -1.0f) ? -1.0f : (fy > 1.0f ? 1.0f : fy);
            float yawR   = atan2f(fx, -fz);
            float pitchR = asinf(pyc);
            // ROLL - fixed 2026-07-22: "look behind and the camera flips upside down".
            //
            // Yaw and pitch above are read off the FORWARD vector, which is the yaw-pitch-roll
            // (Y then X then Z) decomposition. Roll must come from the SAME decomposition, and
            // for R = Ry(yaw)*Rx(pitch)*Rz(roll) that is atan2(R[1][0], R[1][1]):
            //     R[1][0] = 2(xy + wz)        R[1][1] = 1 - 2(x^2 + z^2)
            // The old denominator was 1 - 2(y^2 + z^2), which is R[0][0] - the roll term of a
            // DIFFERENT euler sequence. Substituting a pure yaw of theta makes that denominator
            // cos(theta): it goes negative past 90 degrees, so atan2(0, negative) snaps roll to
            // exactly 180 degrees and the whole view turns over the moment you look behind you.
            // R[1][1] stays +1 through a full turn, so this is stable all the way round.
            float rollR  = atan2f(2.0f * (q.x * q.y + q.w * q.z),
                                  1.0f - 2.0f * (q.x * q.x + q.z * q.z));

            // (Smoothing already applied to the quaternion above, so the angles here are the
            // smoothed ones; nothing further to filter.)
            // [POSEFREEZE] Under the worker these are the PENDING angles - they reach the game
            // only at the next publish, frozen for that whole frame. Inline mode writes the
            // live values directly, exactly as before.
            if (!g_useWorker) { g_olYawRad = yawR; g_olPitchRad = pitchR; g_olRollRad = rollR; }
            else { g_pendYaw = yawR; g_pendPitch = pitchR; g_pendRoll = rollR; }
            // Rebuild the declared orientation from the SAME three numbers the camera hook is
            // about to use. Two reasons it is rebuilt rather than passed through:
            //  - head tilt must do nothing, so roll is omitted unless HeadRoll is on. Claim roll
            //    the camera did not render and the compositor rotates the flat eye image to
            //    reconcile them - the picture spins with empty wedges at the corners.
            //  - smoothing changed the angles above; the declared pose has to carry the smoothed
            //    ones or the compositor is reconciling against a frame that was never rendered.
            // Ry(yaw) * Rx(pitch) [* Rz(roll)] is the composition the extraction decomposes, and
            // g_olYawRad is the NEGATED xr yaw. rolltest.cpp covers this round trip.
            {
                const float hy = -yawR * 0.5f, hp = pitchR * 0.5f;
                const XrQuaternionf qy = { 0.0f, sinf(hy), 0.0f, cosf(hy) };
                const XrQuaternionf qx = { sinf(hp), 0.0f, 0.0f, cosf(hp) };
                q = QMul(qy, qx);
                if (OLUE3_HeadRollEnabled()) {
                    const float hr = rollR * 0.5f;
                    const XrQuaternionf qz = { 0.0f, 0.0f, sinf(hr), cosf(hr) };
                    q = QMul(q, qz);
                }
            }
            // Remember the HEADSET's own field of view. The flat/quad path used to guess a fixed
            // 75 deg, which on a ~100 deg headset leaves a band of black down both sides - the
            // "menu doesn't fill the eye" complaint. The runtime provides the real number; use it.
            g_rtFovX = av[0].fov.angleRight - av[0].fov.angleLeft;
            g_rtFovY = av[0].fov.angleUp    - av[0].fov.angleDown;
            // [XRRUNTIME] keep each eye's own asymmetric frustum - the crop below needs the real
            // per-eye numbers, not the symmetric width above.
            g_rtEyeFov[0] = av[0].fov; g_rtEyeFov[1] = av[1].fov; g_rtEyeFovValid = true;
            if (g_fakeEyeFov) {
                // [Debug] FakeEyeFov: Quest 3 through Virtual Desktop (-54/+40, +44/-55 degrees,
                // mirrored for the right eye), so a symmetric test headset exercises [LENSFIT].
                const float d2r = 0.0174532925f;
                g_rtEyeFov[0] = { -54.0f * d2r, 40.0f * d2r, 44.0f * d2r, -55.0f * d2r };
                g_rtEyeFov[1] = { -40.0f * d2r, 54.0f * d2r, 44.0f * d2r, -55.0f * d2r };
            }
            // [XRSUBMIT] keep the runtime's OWN per-eye poses too. The gap between them is the
            // IPD the runtime expects the two images to have been rendered with - and this mod
            // deliberately declares ONE centre pose for both eyes because the stereo lives in
            // the pixels. A runtime that reprojects each view to its own eye point would then
            // add its separation on top of this mod's = over-separated, unfusable = "double".
            // This line is what distinguishes that from a leftover FOV error.
            g_rtEyePose[0] = av[0].pose; g_rtEyePose[1] = av[1].pose;
            {
                const float ex = av[1].pose.position.x - av[0].pose.position.x;
                const float ey = av[1].pose.position.y - av[0].pose.position.y;
                const float ez = av[1].pose.position.z - av[0].pose.position.z;
                g_rtEyeSep = sqrtf(ex * ex + ey * ey + ez * ez);
            }
            // [HANDSYNC] The hands for THIS camera pose: the same display time (one period ahead,
            // see the pose note above), the same recenter frame. Located here, outside the lock,
            // and stored beside the head below so both promote together.
            OLXrFrameHands fh = {};
            fh.headRaw[0] = rawRelP.x; fh.headRaw[1] = rawRelP.y; fh.headRaw[2] = rawRelP.z;
            if (g_inputReady && g_fn.locateSpace) {
                for (int i = 0; i < 2; ++i) {
                    if (!g_handSpace[i]) continue;
                    XrSpaceLocation hl = {}; hl.type = XR_TYPE_SPACE_LOCATION_VALUE;
                    if (!XrSucceeded(g_fn.locateSpace(g_handSpace[i], g_localSpace, vli.displayTime, &hl))) continue;
                    if (!(hl.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT_VALUE) ||
                        !(hl.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT_VALUE)) continue;
                    const XrVector3f hd = { hl.pose.position.x - g_baseP.x,
                                            hl.pose.position.y - g_baseP.y,
                                            hl.pose.position.z - g_baseP.z };
                    const XrVector3f hr = QRotate(QConj(g_baseQ), hd);
                    const XrQuaternionf hq = QMul(QConj(g_baseQ), QNormalize(hl.pose.orientation));
                    fh.hand[i] = { hq.x, hq.y, hq.z, hq.w, hr.x, hr.y, hr.z };
                    fh.valid |= (1u << i);
                }
            }
            g_devHeadRel = rawRelP;
            if (g_devFakeHands) DevFakeHands(fh.hand, &fh.valid, vli.displayTime);

            // [POSEFREEZE] Same split for the pose itself. The lock guards the struct copy: the
            // game thread reads it at publish, and a torn 7-float read is its own jitter source.
            AcquireSRWLockExclusive(&g_pubLock);
            if (!g_useWorker) {
                g_headPoseApp.orientation = q; g_headPoseApp.position = relP;
                g_olHeadX = relP.x; g_olHeadY = relP.y; g_olHeadZ = relP.z;
                g_olPoseValid = 1;
                g_liveHands = fh; g_liveHandsValid = 1;
                g_rwLat[0] = g_rwCum[0]; g_rwLat[1] = g_rwCum[1];    // [ROOMWALK]
            } else {
                g_pendPose.orientation = q; g_pendPose.position = relP;
                g_pendValid = 1;
                g_pendHands = fh; g_pendHandsValid = 1;
                g_rwPend[0] = g_rwCum[0]; g_rwPend[1] = g_rwCum[1];  // [ROOMWALK] the steps this pose includes
            }
            ReleaseSRWLockExclusive(&g_pubLock);
        }
    }

    int layerCount = 0;
    XrCompositionLayerProjection proj = {};
    XrCompositionLayerProjectionView pv[2] = {};
    XrCompositionLayerQuad quad = {};
    XrCompositionLayerQuad brandQuad = {};
    XrCompositionLayerQuad eyeQuad[2] = {};      // cinema screen: one quad per eye
    bool sbsProjectionSubmitted = false;                     // [STEAMVRSCENE] see below
    XrCompositionLayerProjection blackProj = {};             // [STEAMVRSCENE] placeholder, see below
    XrCompositionLayerProjectionView blackPv[2] = {};
    const XrCompositionLayerBaseHeader* layers[20];           // black placeholder + room + plaque + two eye screens + [MARKERS]
    XrCompositionLayerQuad mkQuad[kMaxMarkers] = {};          // [MARKERS] live until xrEndFrame
    XrCompositionLayerQuad fsQuad = {};                        // [FLIPSCREEN] (Test43) live until xrEndFrame
    bool vfShown = false;                                     // [VIEWFINDER] this frame went to the viewfinder
    TheaterLayers th = {};                                    // [THEATER] lives until xrEndFrame
    bool forceBlack = false;                                  // [FADE] the cut frame of a presentation change

    // ---- SBS path: the engine split the frame, so each half is one eye ----
    bool sbs = false;
    const int mode = OLUE3_StereoMode();
    if (fs.shouldRender && OLUE3_SplitActive() && EnsureEyeSwapchains()) {
        bool eyesOk = false;
        // The angle the rendered picture covers has to be known BEFORE the blit in [FITFRUSTUM] mode,
        // because it decides where inside the runtime's eye the picture is placed.
        float fx = 0.0f, fy = 0.0f;
        if (dev) OLUE3_GetEyeFov(&fx, &fy); else { fx = g_conFovX; fy = g_conFovY; }
        // [LENSFIT] this image was rendered with each eye's own frustum: map it 1:1.
        // [VIEWFINDER] (Test37) or it is a camcorder-zoom frame for the head-locked viewfinder.
        const long frameFlags = dev ? (OLUE3_LensFitLive() | (OLInteract_ViewfinderZoom() > 0.0f ? 2 : 0)) : g_conLensFit;
        const bool lensFrame = (frameFlags & 1) != 0;
        const bool vfFrame = (frameFlags & 2) != 0;
        // [THEATER] A narrow picture headed for the theater screen is blitted whole: fitting it into
        // the runtime frustum would leave it a small box in a black field on that screen.
        const float fxDeg = fx * 57.2957795f;
        const bool screenWanted = !vfFrame && g_screenBelowDeg > 0.5f && fx > 0.1f &&
            (g_screenOn ? fxDeg <= g_screenBelowDeg + 3.0f : fxDeg < g_screenBelowDeg - 3.0f);
        const bool fitMode = g_fitFrustum && g_rtEyeFovValid && fx > 0.1f && fy > 0.1f && !screenWanted && !vfFrame;
        bool fittedToRuntime = false;
        // [HELDFRAME] (Test35) When the compositor runs faster than the game, most of its frames
        // show the SAME game image. Redrawing both eyes for those was pure GPU cost (two full-eye
        // shader passes + copies per compositor frame, ~40% of them repeats on the null headset).
        // OpenXR shows a swapchain's most recently released image, so a repeat just submits the
        // layer again. Anything that changes the eye pictures - a new frame, convergence, fade,
        // sharpening, the eye swap, a swapchain rebuild - forces a redraw.
        struct EyeSig { uint64_t seq, scL, scR; float conv, fade, sharp; int swap, up, fit, mode, me, slack; UINT w, h, ew, eh; };
        static EyeSig s_lastSig; static bool s_lastSigValid = false, s_lastFitted = false;
        EyeSig sig; memset(&sig, 0, sizeof(sig));
        sig.seq = g_conSeq; sig.scL = (uint64_t)g_scL; sig.scR = (uint64_t)g_scR;
        sig.conv = g_convShift; sig.fade = g_presentFade; sig.sharp = g_upscaleSharpness;
        sig.swap = OLUE3_GetEyeSwap(); sig.up = g_upscaler; sig.fit = fitMode ? 1 : 0; sig.mode = mode;
        sig.me = (g_meModel ? 1 : 0) | (lensFrame ? 2 : 0) | (vfFrame ? 4 : 0); sig.slack = g_eyeSlack; sig.w = g_w; sig.h = g_h; sig.ew = g_eyeW; sig.eh = g_eyeH;
        const bool held = !dev && preCaptured && s_lastSigValid && memcmp(&sig, &s_lastSig, sizeof(sig)) == 0;
        if (held) {
            eyesOk = true;
            fittedToRuntime = s_lastFitted;
            InterlockedIncrement(&g_heldEyeFrames);
        } else if (mode == 3) {
            // AER: this frame rendered one eye full-screen. Store it, then re-submit both eyes.
            // Dev-only lane: needs the live device pairing, stands down under the worker.
            if (dev && preCaptured && EnsureEyeHold()) {
                const int cur = OLUE3_CurrentEye() ? 1 : 0;
                g_ctx->CopyResource(g_eyeHold[cur], g_uploadTex);
                eyesOk = BlitHoldToEye(g_scL, g_imgL, g_eyeHold[0]) &&
                         BlitHoldToEye(g_scR, g_imgR, g_eyeHold[1]);
            }
        } else if (mode == 2) {
            // SFR: left eye is the copy taken between the two passes, right eye is the backbuffer.
            // Dev-only lane: needs two live-device captures per frame, stands down under the worker.
            IDirect3DSurface9* lsurf = dev ? OLProxy_LeftEye() : nullptr;
            if (lsurf && CaptureSurface(dev, lsurf)) {
                eyesOk = fitMode
                       ? BlitFitToEye(g_scL, g_imgL, 0.0f, (float)g_w, 0.0f, (float)g_w,
                                      fx, fy, g_rtEyeFov[0])
                       : BlitHalfToEye(g_scL, g_imgL, 0.0f, 0.0f, (float)g_w);
            }
            if (eyesOk && CaptureBackbuffer(dev)) {
                eyesOk = fitMode
                       ? BlitFitToEye(g_scR, g_imgR, 0.0f, (float)g_w, 0.0f, (float)g_w,
                                      fx, fy, g_rtEyeFov[1])
                       : BlitHalfToEye(g_scR, g_imgR, 0.0f, 0.0f, (float)g_w);
            } else {
                eyesOk = false;
            }
            fittedToRuntime = fitMode && eyesOk;
        } else if (g_meModel) {
            // [STEREOMODEL 1 - Mass Effect] Sample a shifted window per half with the edge clamped
            // inside that half. Nothing is reserved, so the eye keeps the FULL half at every
            // convergence setting and no field of view is lost. The sign rides the HALF, not the
            // destination eye, exactly like ME: left half -conv, right half +conv.
            // SWAP flips the ROUTING here while ol_ue3 flips the CAMERAS with the same flag. The
            // two cancel for depth - the left eye still sees the left viewpoint, so there is no
            // inversion and no near-object doubling - and what actually changes is that the
            // convergence plane inverts, because the convergence sign rides the half.
            const float halfW  = (float)(g_w / 2);
            const float convPx = g_convShift * kConvUnit * halfW;
            const int   swap   = OLUE3_GetEyeSwap();
            XrSwapchain scL = swap ? g_scR : g_scL;  auto& imL = swap ? g_imgR : g_imgL;
            XrSwapchain scR = swap ? g_scL : g_scR;  auto& imR = swap ? g_imgL : g_imgR;
            g_cropLx = (int)(0.0f - convPx); g_cropRx = (int)(halfW + convPx);
            if (fitMode) {
                // Each half feeds whichever EYE the swap routing sends it to, so it must be
                // fitted to THAT eye's frustum - they are mirror images of each other.
                const int eyeOfLeftHalf  = swap ? 1 : 0;
                const int eyeOfRightHalf = swap ? 0 : 1;
                if (lensFrame) {
                    // [LENSFIT] each half already covers exactly its eye's frustum.
                    const XrFovf& fl = g_rtEyeFov[eyeOfLeftHalf];
                    const XrFovf& fr = g_rtEyeFov[eyeOfRightHalf];
                    eyesOk = preCaptured
                          && BlitFitToEyeTan(scL, imL, 0.0f - convPx, halfW - convPx, 0.0f, halfW,
                                             tanf(fl.angleLeft), tanf(fl.angleRight),
                                             tanf(fl.angleUp), tanf(fl.angleDown), fl)
                          && BlitFitToEyeTan(scR, imR, halfW + convPx, (float)g_w + convPx, halfW, (float)g_w,
                                             tanf(fr.angleLeft), tanf(fr.angleRight),
                                             tanf(fr.angleUp), tanf(fr.angleDown), fr);
                } else {
                    eyesOk = preCaptured
                          && BlitFitToEye(scL, imL, 0.0f - convPx, halfW - convPx, 0.0f, halfW,
                                          fx, fy, g_rtEyeFov[eyeOfLeftHalf])
                          && BlitFitToEye(scR, imR, halfW + convPx, (float)g_w + convPx, halfW, (float)g_w,
                                          fx, fy, g_rtEyeFov[eyeOfRightHalf]);
                }
                fittedToRuntime = eyesOk;
            }
            if (!fittedToRuntime)
            eyesOk = preCaptured
                  && BlitHalfToEye(scL, imL, 0.0f  - convPx, 0.0f,  halfW)
                  && BlitHalfToEye(scR, imR, halfW + convPx, halfW, (float)g_w);
        } else {
            // Convergence: take each eye's half from a window slid inward/outward by conv pixels.
            // Positive moves the halves TOWARD each other, pushing the zero-parallax plane
            // further away. Clamped so the source window always stays inside the frame - a box
            // that runs off the edge is a silent no-copy in D3D11, i.e. a frozen eye.
            // ME1's sign: the LEFT window slides left and the RIGHT window slides right, so the two
            // images move toward each other and the zero-parallax plane pushes away. Each window is
            // clamped INSIDE its own half, so an eye can never sample the other eye's picture.
            const int halfW = (int)(g_w / 2);
            const int conv  = (int)(g_convShift * kConvUnit * (float)halfW);
            int lx = g_eyeSlack - conv;
            int rx = halfW + g_eyeSlack + conv;
            if (lx < 0) lx = 0;
            if (lx > halfW - (int)g_srcEyeW) lx = halfW - (int)g_srcEyeW;
            if (rx < halfW) rx = halfW;
            if (rx > (int)g_w - (int)g_srcEyeW) rx = (int)g_w - (int)g_srcEyeW;
            g_cropLx = lx; g_cropRx = rx;
            eyesOk = preCaptured
                  && BlitHalfToEye(g_scL, g_imgL, (float)lx, (float)lx, (float)(lx + g_srcEyeW))
                  && BlitHalfToEye(g_scR, g_imgR, (float)rx, (float)rx, (float)(rx + g_srcEyeW));
        }
        if (!held) {
            s_lastSigValid = eyesOk;
            if (eyesOk) { s_lastSig = sig; s_lastFitted = fittedToRuntime; }
        }
        if (eyesOk) {
            // fx/fy were read above ([FOVLATCH]: under the worker these belong to the image on
            // screen, not to whatever the game's camera is doing right now).
            if (vfFrame && fx > 0.1f && fy > 0.1f) {
                // [VIEWFINDER] (Test37) Camcorder zoom: the zoomed picture sits in a head-locked frame
                // of fixed angular size with black around it - holding the camcorder to your eye.
                // Magnification = that size over the angle rendered (ol_ue3 narrows the render with
                // the zoom). The rest of the headset stays dark, so nothing in view moves wrongly.
                const float kVfDeg = 60.0f, kVfDist = 1.0f;
                // (Test46) "Without the black bars is better - it feels weird with them": by default the
                // zoomed picture fills each eye's whole view (the full picture, its own aspect, covering the
                // lens); [Motion] ViewfinderFrame=1 is the framed 60 deg camcorder window with black around.
                const bool vfFill = !OLInteract_Get(12) && g_rtEyeFovValid;
                float qw = 2.0f * kVfDist * tanf(kVfDeg * 0.5f * 0.0174532925f);
                // A camcorder's 4:3 landscape frame: the centre rows of the (taller) eye picture.
                const int32_t cropH = vfFill ? (int32_t)g_eyeH : ((g_eyeH * 4 > g_eyeW * 3) ? (int32_t)(g_eyeW * 3 / 4) : (int32_t)g_eyeH);
                const int32_t cropY = ((int32_t)g_eyeH - cropH) / 2;
                float qh = qw * (tanf(fy * 0.5f) * (float)cropH / (float)g_eyeH) / tanf(fx * 0.5f);
                if (vfFill) {
                    // (Test46c) Both eyes' pictures centred straight ahead, exactly like the framed viewfinder,
                    // just big enough to cover both lenses. The zoomed picture is rendered straight ahead; the
                    // first fill build centred each eye's copy on that eye's lop-sided lens field instead (the
                    // left one pushed left, the right one right) and the eyes could not fuse them - "a double
                    // vision look".
                    float hx = 0.0f, hy = 0.0f;
                    for (int e = 0; e < 2; ++e) {
                        const XrFovf& f = g_rtEyeFov[e];
                        hx = fmaxf(hx, fmaxf(fabsf(tanf(f.angleLeft)), fabsf(tanf(f.angleRight))));
                        hy = fmaxf(hy, fmaxf(fabsf(tanf(f.angleUp)), fabsf(tanf(f.angleDown))));
                    }
                    const float aspect = tanf(fx * 0.5f) / tanf(fy * 0.5f);          // the picture's width over height
                    qw = 2.0f * hx * kVfDist; qh = qw / aspect;
                    if (qh < 2.0f * hy * kVfDist) { qh = 2.0f * hy * kVfDist; qw = qh * aspect; }
                    qw *= 1.03f; qh *= 1.03f;                                          // a hair past the lens edge
                }
                for (int e = 0; e < 2; ++e) {
                    const float qx = 0.0f, qy = 0.0f;
                    eyeQuad[e].type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
                    eyeQuad[e].layerFlags = 0;
                    eyeQuad[e].eyeVisibility = e ? XR_EYE_VISIBILITY_RIGHT_VALUE : XR_EYE_VISIBILITY_LEFT_VALUE;
                    eyeQuad[e].subImage.swapchain = e ? g_scR : g_scL;
                    eyeQuad[e].subImage.imageArrayIndex = 0;
                    eyeQuad[e].subImage.imageRect.offset = { 0, cropY };
                    eyeQuad[e].subImage.imageRect.extent = { (int32_t)g_eyeW, cropH };
                    eyeQuad[e].space = g_viewSpace;
                    eyeQuad[e].pose = IdentityPose(); eyeQuad[e].pose.position.z = -kVfDist;
                    eyeQuad[e].pose.position.x = qx; eyeQuad[e].pose.position.y = qy;
                    eyeQuad[e].size.width = qw; eyeQuad[e].size.height = qh;
                    layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&eyeQuad[e]);
                }
                if (g_screenOn) { InterlockedExchange(&g_screenOn, 0); g_screenPoseValid = 0; }
                sbs = true; vfShown = true;
                static long s_vfLogs = 0;
                if (InterlockedIncrement(&s_vfLogs) <= 3)
                    XLOG("[OLVR][VIEWFINDER] zoomed picture %s: rendered %.1f deg",
                         vfFill ? "filling the view" : "in the 60 deg viewfinder frame", fx * 57.2957795f);
            } else if (fx > 0.1f && fy > 0.1f) {
                // Cinema-screen decision, with a +/-3 deg hysteresis band so a FOVApproach sweep
                // through the threshold flips the presentation once, not every frame.
                const float fovDeg = fx * 57.2957795f;
                if (g_screenBelowDeg > 0.5f) {
                    if (!g_screenOn && fovDeg < g_screenBelowDeg - 3.0f) {
                        InterlockedExchange(&g_screenOn, 1);
                        // Plant the theater screen in the ROOM: kScreenDist in front of where
                        // the head is right now, upright (yaw-only - a pitched/rolled anchor
                        // would hang the screen tilted for the whole cutscene).
                        if (!g_cinemaHeadLocked && g_lastLocalValid) {
                            const XrVector3f fwd = QRotate(g_lastLocalQ, { 0.0f, 0.0f, -1.0f });
                            const float yaw = atan2f(-fwd.x, -fwd.z);
                            // Rotation about +Y by yaw: rotating (0,0,-1) by +theta lands at
                            // (-sin,0,-cos), and atan2(-fwd.x,-fwd.z) recovers exactly +theta -
                            // so the quaternion takes +yaw/2, no negation. (rolltest lesson:
                            // derive the round trip, never eyeball quaternion signs.)
                            const XrQuaternionf yq = { 0.0f, sinf(yaw * 0.5f), 0.0f, cosf(yaw * 0.5f) };
                            const XrVector3f off = QRotate(yq, { 0.0f, 0.0f, -g_cinemaDistanceM });
                            g_screenPose.orientation = yq;
                            g_screenPose.position = { g_lastLocalP.x + off.x,
                                                      g_lastLocalP.y + off.y,
                                                      g_lastLocalP.z + off.z };
                            g_screenPoseValid = 1;
                        } else g_screenPoseValid = 0;
                        XLOG("[OLVR][XR] cinema screen ON (eyeFov %.1f < %.1f deg) - %s", fovDeg,
                             g_screenBelowDeg, g_screenPoseValid ? "world-locked in front of you" : "head-locked (no pose yet)");
                    } else if (g_screenOn && fovDeg > g_screenBelowDeg + 3.0f) {
                        InterlockedExchange(&g_screenOn, 0);
                        g_screenPoseValid = 0;
                        XLOG("[OLVR][XR] cinema screen OFF (eyeFov %.1f > %.1f deg) - back to true projection", fovDeg, g_screenBelowDeg);
                    }
                } else if (g_screenOn) InterlockedExchange(&g_screenOn, 0);

                if (g_screenOn && !g_cinemaHeadLocked && g_lastLocalValid && fy > 0.1f) {
                    // [THEATER] The per-eye stereo images on the theater screen, at the eye image's
                    // true angular aspect (not forced 16:9, which stretched them).
                    const XrSwapchain scs[2] = { g_scL, g_scR };
                    const XrRect2Di rect = { { 0, 0 }, { (int32_t)g_eyeW, (int32_t)g_eyeH } };
                    bool scene = false;
                    layerCount = AddTheaterLayers(th, layers, 0, true, scs, rect,
                                                  tanf(fx * 0.5f) / tanf(fy * 0.5f), &scene);
                    if (scene) sbsProjectionSubmitted = true;
                    sbs = true;
                } else if (g_screenOn) {
                    // Per-eye quads: LEFT eye sees the left render, RIGHT eye the right render, on
                    // one coplanar screen grown to the headset's real FOV keeping the eye image's
                    // own aspect (stretching is the "squashed" look; never do it). WORLD-locked at
                    // the anchor captured when the screen engaged - a theater screen fixed in the
                    // room - falling back to head-locked only if no pose has ever been located.
                    const float kDist = g_cinemaDistanceM;
                    const float hfx = (g_rtFovX > 0.2f) ? g_rtFovX : 1.75f;   // ~100 deg fallback
                    const float hfy = (g_rtFovY > 0.2f) ? g_rtFovY : 1.75f;
                    const float maxW = 2.0f * kDist * tanf(hfx * 0.5f) * 0.98f;
                    const float maxH = 2.0f * kDist * tanf(hfy * 0.5f) * 0.98f;
                    const float wantW = 2.0f * kDist * tanf(g_cinemaWidthDeg * 0.5f * 0.0174532925f);
                    // This path is only for narrow 2D presentation. Each SBS eye texture is stored
                    // in a half-width render target, but the authored menu/logo itself is 16:9;
                    // using the storage aspect here makes the theater screen unnaturally tall.
                    float qw = wantW < maxW ? wantW : maxW, qh = qw * (9.0f / 16.0f);
                    if (qh > maxH) { qh = maxH; qw = qh * (16.0f / 9.0f); }
                    int layerBase = 0;
                    if (EnsureBrandScene()) {
                        brandQuad.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
                        brandQuad.layerFlags = 0; brandQuad.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
                        brandQuad.subImage.swapchain = g_brandSc; brandQuad.subImage.imageArrayIndex = 0;
                        brandQuad.subImage.imageRect.offset = { 0, 0 };
                        brandQuad.subImage.imageRect.extent = { kBrandW, kBrandH };
                        brandQuad.space = g_screenPoseValid ? g_localSpace : g_viewSpace;
                        brandQuad.pose = g_screenPoseValid ? g_screenPose : IdentityPose();
                        if (!g_screenPoseValid) brandQuad.pose.position.z = -kDist;
                        brandQuad.size.width = qw + 0.30f;
                        brandQuad.size.height = brandQuad.size.width * (float)kBrandH / (float)kBrandW;
                        layers[layerBase++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&brandQuad);
                    }
                    for (int e = 0; e < 2; ++e) {
                        eyeQuad[e].type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
                        eyeQuad[e].layerFlags = 0;
                        eyeQuad[e].eyeVisibility = (e == 0) ? XR_EYE_VISIBILITY_LEFT_VALUE
                                                            : XR_EYE_VISIBILITY_RIGHT_VALUE;
                        eyeQuad[e].subImage.swapchain = (e == 0) ? g_scL : g_scR;
                        eyeQuad[e].subImage.imageArrayIndex = 0;
                        eyeQuad[e].subImage.imageRect.offset = { 0, 0 };
                        eyeQuad[e].subImage.imageRect.extent = { (int32_t)g_eyeW, (int32_t)g_eyeH };
                        if (g_screenPoseValid) {
                            eyeQuad[e].space = g_localSpace;
                            eyeQuad[e].pose  = g_screenPose;
                            const XrVector3f lift = QRotate(g_screenPose.orientation, { 0.0f, 0.16f, 0.025f });
                            eyeQuad[e].pose.position.x += lift.x;
                            eyeQuad[e].pose.position.y += lift.y;
                            eyeQuad[e].pose.position.z += lift.z;
                        } else {
                            eyeQuad[e].space = g_viewSpace;
                            eyeQuad[e].pose  = IdentityPose(); eyeQuad[e].pose.position.y = 0.16f; eyeQuad[e].pose.position.z = -kDist + 0.025f;
                        }
                        eyeQuad[e].size.width = qw; eyeQuad[e].size.height = qh;
                        layers[layerBase + e] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&eyeQuad[e]);
                    }
                    layerCount = layerBase + 2;
                    sbs = true;
                } else {
                for (int e = 0; e < 2; ++e) {
                    pv[e].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW_VALUE;
                    // BOTH eyes carry the SAME (centre head) pose on purpose. The stereo lives in
                    // the IMAGES - the engine already rendered them from two separated cameras.
                    // Giving each eye its own IPD-apart pose makes the compositor reproject them
                    // to slightly different predicted poses, and they diverge during motion.
                    pv[e].pose = poseValidForThisImage ? RecenterToLocal(poseForThisImage) : IdentityPose();   // [LINKPOSE]
                    // [PEREYEPOSE] Opt-in ([XR] PerEyePose=1). The default declares ONE centre
                    // pose for both views because the separation is already in the pixels, and
                    // that is what VDXR and SteamVR were tuned against. A runtime that instead
                    // reprojects each view to its OWN eye point sees two views claiming the same
                    // viewpoint and pulls them apart itself, adding its IPD on top of this mod's -
                    // over-separated and unfusable, i.e. "double everywhere". This declares each
                    // view at the point its image was actually rendered from: the camera used
                    // separation (halfEyeUU / WorldToMeters) along the declared right vector.
                    if (g_perEyePose && g_rtEyeSep > 0.001f) {
                        // Use the runtime's own MEASURED eye separation, not halfEyeUU /
                        // WorldToMeters: that world scale is a dead control in this build
                        // (positional tracking is off, the slider is inert) and at a saved
                        // 3.753uu / 50 it would declare 0.15m - twice the truth, which makes
                        // the doubling worse rather than better. rtEyeSep is what the runtime
                        // itself reports and is what it will reproject to.
                        const float s = (e == 0) ? -g_rtEyeSep * 0.5f : g_rtEyeSep * 0.5f;
                        const XrVector3f right = QRotate(pv[e].pose.orientation, { 1.0f, 0.0f, 0.0f });
                        pv[e].pose.position.x += right.x * s;
                        pv[e].pose.position.y += right.y * s;
                        pv[e].pose.position.z += right.z * s;
                    }
                    // The declared frustum is the CENTERED window, while the copy above took the
                    // SHIFTED window - deliberately, and the distinction is the whole feature
                    // (falsified live 2026-07-23): the crop being NARROWER than the half is
                    // honest and is compensated here through tan space; but the convergence
                    // SHIFT must NOT be compensated. Declaring the shifted crop's true angles
                    // puts every pixel back at its original direction and cancels the depth
                    // change EXACTLY - a mathematical no-op that leaves only the black edge
                    // bars, which is precisely what "convergence does nothing but squash the
                    // image" was. ME1 shifts the image and declares an unshifted frustum; the
                    // ~1-3 deg angular lie IS the toe-in mechanism. The half spans -T..+T in
                    // tan space, an edge at fraction f sits at tan = -T + 2*T*f.
                    if (g_meModel) {
                        // [STEREOMODEL 1] Nothing was reserved, so the image IS the full half and
                        // the honest frustum is the plain symmetric one. The convergence shift is
                        // still deliberately NOT compensated - that angular lie is the toe-in.
                        pv[e].fov.angleLeft  = -fx * 0.5f; pv[e].fov.angleRight = fx * 0.5f;
                    } else {
                        const float halfWf = (float)(g_w / 2);
                        const float f0 = (float)g_eyeSlack / halfWf;
                        const float f1 = f0 + (float)g_srcEyeW / halfWf;
                        const float T  = tanf(fx * 0.5f);
                        pv[e].fov.angleLeft  = atanf(-T + 2.0f * T * f0);
                        pv[e].fov.angleRight = atanf(-T + 2.0f * T * f1);
                    }
                    pv[e].fov.angleUp   =  fy * 0.5f; pv[e].fov.angleDown  = -fy * 0.5f;
                    pv[e].subImage.imageArrayIndex = 0;
                    pv[e].subImage.imageRect.offset = { 0, 0 };
                    pv[e].subImage.imageRect.extent = { (int32_t)g_eyeW, (int32_t)g_eyeH };

                    // [XRRUNTIME] Meta and SteamVR only: declare the INTERSECTION of the declared window and
                    // the runtime's own per-eye frustum, and crop the submitted rect to exactly
                    // that. Pure rect arithmetic in tan space - nothing is re-rendered, and the
                    // declared FOV still describes precisely the pixels being handed over, because
                    // the rect shrinks along with it. Ported from ME1's [LINKFOV]/[STEAMVRFOV].
                    // On Virtual Desktop and every other runtime this never runs.
                    if (fittedToRuntime) {
                        // [FITFRUSTUM] The image already spans this eye exactly, so declare the
                        // runtime's own frustum and hand over the whole thing. Nothing here can
                        // disagree with the pixels, which is the entire point - a runtime that
                        // ignores the declaration lands on the same answer as one that honours it.
                        pv[e].fov = g_rtEyeFov[e];
                        pv[e].subImage.imageRect.offset = { 0, 0 };
                        pv[e].subImage.imageRect.extent = { (int32_t)g_eyeW, (int32_t)g_eyeH };
                    } else if ((g_isOculusRuntime || g_isSteamVrRuntime) && g_rtEyeFovValid) {
                        const XrFovf d = pv[e].fov;              // what was rendered
                        const XrFovf& t = g_rtEyeFov[e];         // what the runtime expects
                        const float aL = (d.angleLeft  > t.angleLeft)  ? d.angleLeft  : t.angleLeft;
                        const float aR = (d.angleRight < t.angleRight) ? d.angleRight : t.angleRight;
                        const float aU = (d.angleUp    < t.angleUp)    ? d.angleUp    : t.angleUp;
                        const float aD = (d.angleDown  > t.angleDown)  ? d.angleDown  : t.angleDown;
                        const float tanDL = tanf(d.angleLeft), tanDR = tanf(d.angleRight);
                        const float tanDU = tanf(d.angleUp),   tanDD = tanf(d.angleDown);
                        const float hSpan = tanDR - tanDL, vSpan = tanDU - tanDD;
                        if (aR - aL > 0.05f && aU - aD > 0.05f && hSpan > 1e-4f && vSpan > 1e-4f) {
                            const int32_t w = (int32_t)g_eyeW, h = (int32_t)g_eyeH;
                            int32_t x0 = (int32_t)lroundf((tanf(aL) - tanDL) / hSpan * w);
                            int32_t x1 = (int32_t)lroundf((tanf(aR) - tanDL) / hSpan * w);
                            int32_t y0 = (int32_t)lroundf((tanDU - tanf(aU)) / vSpan * h);
                            int32_t y1 = (int32_t)lroundf((tanDU - tanf(aD)) / vSpan * h);
                            if (x0 < 0) x0 = 0; if (x0 > w - 1) x0 = w - 1;
                            if (x1 < x0 + 1) x1 = x0 + 1; if (x1 > w) x1 = w;
                            if (y0 < 0) y0 = 0; if (y0 > h - 1) y0 = h - 1;
                            if (y1 < y0 + 1) y1 = y0 + 1; if (y1 > h) y1 = h;
                            pv[e].subImage.imageRect.offset = { x0, y0 };
                            pv[e].subImage.imageRect.extent = { x1 - x0, y1 - y0 };
                            pv[e].fov.angleLeft = aL; pv[e].fov.angleRight = aR;
                            pv[e].fov.angleUp   = aU; pv[e].fov.angleDown  = aD;
                            static int s_cropLogs = 0;
                            if (s_cropLogs < 2) { ++s_cropLogs;
                                XLOG("[OLVR][XRRUNTIME] eye %d cropped to the runtime frustum: "
                                     "rect %dx%d+%d,%d fov %.1f/%.1f deg horizontal",
                                     e, x1 - x0, y1 - y0, x0, y0,
                                     aL * 57.2957795f, aR * 57.2957795f); }
                        }
                    }
                }
                pv[0].subImage.swapchain = g_scL;
                pv[1].subImage.swapchain = g_scR;
                // [FLIPSCREEN] (Test43) the left eye's view, for where the camcorder's lens points in it
                // the lens maths lives in the recenter frame, so it gets the pose before [LINKPOSE]
                g_fsViewQ = poseValidForThisImage ? poseForThisImage.orientation : IdentityPose().orientation;
                g_fsLens = lensFrame && fittedToRuntime; g_fsFx = fx; g_fsFy = fy;
                for (int e = 0; e < 2; ++e) {
                    g_dbgDeclFov[e] = pv[e].fov;
                    g_dbgRect[e][0] = pv[e].subImage.imageRect.offset.x;
                    g_dbgRect[e][1] = pv[e].subImage.imageRect.offset.y;
                    g_dbgRect[e][2] = pv[e].subImage.imageRect.extent.width;
                    g_dbgRect[e][3] = pv[e].subImage.imageRect.extent.height;
                }
                g_dbgDeclPose = pv[0].pose;
                proj.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VALUE;
                proj.layerFlags = 0; proj.space = g_appSpace; proj.viewCount = 2; proj.views = pv;
                sbsProjectionSubmitted = true;   // [STEAMVRSCENE] a real scene layer went out
                layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&proj);
                layerCount = 1;
                sbs = true;
                static bool once = false;
                if (!once) { once = true;
                    XLOG("[OLVR][XR] STEREO submit live: %ux%u per eye, fov %.1f x %.1f deg",
                         g_eyeW, g_eyeH, fx * 57.2957795f, fy * 57.2957795f); }
                }
            }
        }
    }

    if (!sbs && fs.shouldRender && preCaptured && BlitToSwapchain()) {
        // Until the camera lane reports the game's REAL rendered FOV, the quad is the honest
        // presentation: a flat panel makes no claim about world geometry. Projection (P) is a
        // debug/preview path and WILL be wrong-scaled until FOV is wired.
        float fovX = (g_olDeclFovXDeg > 0.5f) ? g_olDeclFovXDeg * 0.01745329f : 1.5708f;  // 90deg = Outlast DefaultFOV
        float fovY = (g_olDeclFovYDeg > 0.5f) ? g_olDeclFovYDeg * 0.01745329f
                                              : 2.0f * atanf(tanf(fovX * 0.5f) * (float)g_h / (float)g_w);
        if (g_olUseProjection) {
            for (int e = 0; e < 2; ++e) {
                pv[e].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW_VALUE;
                pv[e].pose = g_olPoseValid ? RecenterToLocal(g_headPoseApp) : IdentityPose();
                pv[e].fov.angleLeft = -fovX * 0.5f; pv[e].fov.angleRight = fovX * 0.5f;
                pv[e].fov.angleUp = fovY * 0.5f;    pv[e].fov.angleDown = -fovY * 0.5f;
                pv[e].subImage.swapchain = g_swapchain;      // mono: both eyes read the same image
                pv[e].subImage.imageArrayIndex = 0;
                pv[e].subImage.imageRect.offset = { 0, 0 };
                pv[e].subImage.imageRect.extent = { (int32_t)g_w, (int32_t)g_h };
            }
            proj.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VALUE;
            proj.layerFlags = 0; proj.space = g_appSpace; proj.viewCount = 2; proj.views = pv;
            sbsProjectionSubmitted = true;   // [STEAMVRSCENE] mono projection is a scene layer too
            layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&proj);
            layerCount = 1;
        } else {
            // The quad is a virtual SCREEN showing the full frame (menus, loading). Its width
            // used to follow the last captured per-eye world FOV, so after a narrow-FOV stretch
            // (camcorder zoom, intro cameras at ~33deg) the main menu inherited that angle and
            // showed as a postage stamp floating in black. A flat screen makes no claim about
            // world geometry - give it a fixed, comfortable width instead.
            // Fill as much of the headset as the image honestly can. Grow the quad to the runtime's
            // real FOV, keep the frame's own aspect (never stretch - that is the "squashed" look),
            // and if that makes it taller than the headset, fit to the height instead. Whatever
            // black is left over is the difference between a 16:9 frame and a near-square lens,
            // and the only way to remove THAT is to render wider, not to distort the picture that exists.
            if (!g_cinemaHeadLocked && g_lastLocalValid) {
                // [THEATER] The main menu, logos and loading screens: world-locked screen in the room.
                const XrSwapchain scs[2] = { g_swapchain, g_swapchain };
                const XrRect2Di rect = { { 0, 0 }, { (int32_t)g_w, (int32_t)g_h } };
                bool scene = false;
                layerCount = AddTheaterLayers(th, layers, 0, false, scs, rect, (float)g_w / (float)g_h, &scene);
                if (scene) sbsProjectionSubmitted = true;
            } else {
            const float kDist = g_cinemaDistanceM;
            const float fovX = (g_rtFovX > 0.2f) ? g_rtFovX : 1.75f;   // ~100 deg fallback
            const float fovY = (g_rtFovY > 0.2f) ? g_rtFovY : 1.75f;
            const float maxW = 2.0f * kDist * tanf(fovX * 0.5f) * 0.98f;
            const float maxH = 2.0f * kDist * tanf(fovY * 0.5f) * 0.98f;
            const float wantW = 2.0f * kDist * tanf(g_cinemaWidthDeg * 0.5f * 0.0174532925f);
            float qw = wantW < maxW ? wantW : maxW, qh = qw * (float)g_h / (float)g_w;
            if (qh > maxH) { qh = maxH; qw = qh * (float)g_w / (float)g_h; }
            quad.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE; quad.layerFlags = 0; quad.space = g_viewSpace;
            quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
            quad.subImage.swapchain = g_swapchain; quad.subImage.imageArrayIndex = 0;
            quad.subImage.imageRect.offset = { 0, 0 };
            quad.subImage.imageRect.extent = { (int32_t)g_w, (int32_t)g_h };
            quad.pose = IdentityPose(); quad.pose.position.y = 0.16f; quad.pose.position.z = -kDist + 0.025f;
            quad.size.width = qw; quad.size.height = qh;
            int layerBase = 0;
            if (EnsureBrandScene()) {
                brandQuad.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE; brandQuad.layerFlags = 0;
                brandQuad.space = g_viewSpace; brandQuad.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
                brandQuad.subImage.swapchain = g_brandSc; brandQuad.subImage.imageArrayIndex = 0;
                brandQuad.subImage.imageRect.offset = { 0, 0 };
                brandQuad.subImage.imageRect.extent = { kBrandW, kBrandH };
                brandQuad.pose = IdentityPose(); brandQuad.pose.position.z = -kDist;
                brandQuad.size.width = qw + 0.30f;
                brandQuad.size.height = brandQuad.size.width * (float)kBrandH / (float)kBrandW;
                layers[layerBase++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&brandQuad);
            }
            layers[layerBase] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
            layerCount = layerBase + 1;
            }
        }
    }

    // [FADE] A change of presentation (loading panel <-> theater <-> stereo gameplay) cuts to black
    // for this frame and fades the new picture in over ~0.35 s. Leaving a presentation that lasted
    // under 0.3 s does not cut again, so the way BACK from a brief hiccup never strobes.
    {
        // A frame with no layers (nothing to show yet) is not a presentation, so it neither starts
        // a fade nor resets the timer of the one in progress.
        const int modeNow = layerCount == 0 ? 0 : (sbs ? (g_screenOn ? 2 : 3) : 1);
        const DWORD nowMs = GetTickCount();
        static int s_mode = 0;
        static DWORD s_since = 0, s_last = 0;
        if (modeNow > 0 && modeNow != s_mode) {
            if (s_mode > 0 && (nowMs - s_since) > 300u) { g_presentFade = 0.0f; forceBlack = true; }
            s_mode = modeNow; s_since = nowMs;
        }
        float dtf = s_last ? (float)(nowMs - s_last) * 0.001f : 0.0f;
        s_last = nowMs;
        if (dtf > 0.1f) dtf = 0.1f;
        g_presentFade += dtf / 0.35f;
        if (g_presentFade > 1.0f) g_presentFade = 1.0f;
    }

    // [STEAMVRSCENE] SteamVR's compositor is a scene-app state machine, not a layer blender. Quad
    // layers arrive as OVERLAYS, but its waiting room stays up until the app submits PROJECTION
    // frames, and it comes back the moment they stop - so on SteamVR every frame that carries only
    // quads (menus, the cinema screen, anything before the first capture) would show the void with
    // the settings menu floating in it. ME1 hit exactly this and its cure is a tiny opaque-black projection
    // layer slipped UNDERNEATH the quads to keep the scene state alive. It is 64px, it is declared
    // at the runtime's own frustum so it can never trip the crop above, and it is never built or
    // submitted on any other runtime.
    if (g_isSteamVrRuntime && g_blackAllowed && layerCount > 0 && !sbsProjectionSubmitted && g_rtEyeFovValid &&
        EnsureBlackScene()) {
        for (int e = 0; e < 2; ++e) {
            blackPv[e].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW_VALUE;
            blackPv[e].pose = g_olPoseValid ? RecenterToLocal(g_headPoseApp) : IdentityPose();
            blackPv[e].fov  = g_rtEyeFov[e];
            blackPv[e].subImage.swapchain = g_blackSc;
            blackPv[e].subImage.imageArrayIndex = 0;
            blackPv[e].subImage.imageRect.offset = { 0, 0 };
            blackPv[e].subImage.imageRect.extent = { kBlackSceneSize, kBlackSceneSize };
        }
        blackProj.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VALUE;
        blackProj.layerFlags = 0; blackProj.space = g_appSpace;
        blackProj.viewCount = 2; blackProj.views = blackPv;
        // Slide the existing layers up and put the placeholder at the bottom of the stack.
        for (int i = layerCount; i > 0; --i) layers[i] = layers[i - 1];
        layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&blackProj);
        ++layerCount;
        static bool onceBlack = false;
        if (!onceBlack) { onceBlack = true;
            XLOG("[OLVR][XRRUNTIME] SteamVR: adding the black scene layer under quad-only frames"); }
    }

    // [SNAPTURN] Blink: ONLY the 64px black projection, on every runtime. A projection layer (not
    // zero layers) keeps SteamVR's scene state alive and gives Meta's compositor a real black
    // frame to hold, so neither drops to its own void or reprojects the turning world.
    const ULONGLONG blinkUntil = g_blinkUntilMs;
    if ((forceBlack || (blinkUntil && GetTickCount64() < blinkUntil)) && g_blackAllowed &&
        g_rtEyeFovValid && EnsureBlackScene()) {
        for (int e = 0; e < 2; ++e) {
            blackPv[e].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW_VALUE;
            blackPv[e].pose = g_olPoseValid ? RecenterToLocal(g_headPoseApp) : IdentityPose();
            blackPv[e].fov  = g_rtEyeFov[e];
            blackPv[e].subImage.swapchain = g_blackSc;
            blackPv[e].subImage.imageArrayIndex = 0;
            blackPv[e].subImage.imageRect.offset = { 0, 0 };
            blackPv[e].subImage.imageRect.extent = { kBlackSceneSize, kBlackSceneSize };
        }
        blackProj.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VALUE;
        blackProj.layerFlags = 0; blackProj.space = g_appSpace;
        blackProj.viewCount = 2; blackProj.views = blackPv;
        layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&blackProj);
        layerCount = 1;
    }

    // [MARKERS] (Test37) developer markers over ordinary stereo gameplay only - not on the theater
    // screen, the viewfinder, or a blink/cut frame (that one is the black layer alone).
    const bool blackOnly = layerCount == 1 && layers[0] == reinterpret_cast<const XrCompositionLayerBaseHeader*>(&blackProj);
    // [FLIPSCREEN] (Test43) the camcorder's live screen, over ordinary stereo gameplay only
    if (sbs && !vfShown && !g_screenOn && !blackOnly && layerCount > 0 && layerCount < 18 && g_lastLocalValid && FlipScreenLayer(fsQuad))
        layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&fsQuad);
    if (sbs && !vfShown && !g_screenOn && !blackOnly && layerCount > 0 && g_lastLocalValid) {
        OLMarker mk[kMaxMarkers]; int n = 0;
        AcquireSRWLockShared(&g_mkLock);
        n = g_mkN; if (n) memcpy(mk, g_mk, sizeof(OLMarker) * n);
        ReleaseSRWLockShared(&g_mkLock);
        if (n && EnsureMarkers()) {
            // Quest runtimes compose at most 16 layers: the scene plus at most 12 of these.
            const int layerCap = layerCount + 12 < 20 ? layerCount + 12 : 20;
            for (int i = 0; i < n && layerCount < layerCap; ++i) {
                const int c = (mk[i].color >= 0 && mk[i].color < OLMK_COUNT) ? mk[i].color : OLMK_WHITE;
                if (!g_mkAtlas) continue;
                const XrVector3f r = QRotate(g_baseQ, { mk[i].pos[0], mk[i].pos[1], mk[i].pos[2] });
                XrCompositionLayerQuad& q = mkQuad[i];
                q.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
                q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT_VALUE |
                               XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT_VALUE;
                q.space = g_localSpace; q.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
                q.subImage.swapchain = g_mkAtlas; q.subImage.imageArrayIndex = 0;
                q.subImage.imageRect.offset = { g_mkRect[c][0], g_mkRect[c][1] }; q.subImage.imageRect.extent = { g_mkRect[c][2], g_mkRect[c][3] };
                q.pose.orientation = g_lastLocalQ;   // billboard: always faces you
                q.pose.position = { g_baseP.x + r.x, g_baseP.y + r.y, g_baseP.z + r.z };
                q.size.width = mk[i].size; q.size.height = mk[i].aspect > 0.01f ? mk[i].size / mk[i].aspect : mk[i].size;
                layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q);
            }
        }
    }

    // (Beta 1.1) [BATGAUGE] the camcorder is up: its battery, large, low in the view (head-locked), with the
    // spare batteries beside it. Over the stereo picture and over the zoomed viewfinder alike.
    XrCompositionLayerQuad batQuad[2] = {};
    {
        float energy = 1.0f; int spares = 0;
        if (fs.shouldRender && !g_screenOn && !blackOnly && layerCount > 0 && layerCount < 18 &&
            OLInteract_BatteryGauge(&energy, &spares) && EnsureMarkers() && g_mkAtlas) {
            int bars = (int)ceilf(energy * 4.0f - 0.05f);
            if (bars < 0) bars = 0; if (bars > 4) bars = 4;
            if (spares < 0) spares = 0; if (spares > 9) spares = 9;
            const float w = 0.066f, h = w * (float)kBatH / (float)kBatW;      // metres, at 0.55 m: ~7 deg wide (was 0.075)
            for (int k = 0; k < 2; ++k) {
                XrCompositionLayerQuad& q = batQuad[k];
                q.type = XR_TYPE_COMPOSITION_LAYER_QUAD_VALUE;
                q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT_VALUE |
                               XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT_VALUE;
                q.space = g_viewSpace; q.eyeVisibility = XR_EYE_VISIBILITY_BOTH_VALUE;
                q.subImage.swapchain = g_mkAtlas; q.subImage.imageArrayIndex = 0;
                const int* rc = k == 0 ? g_batRect[bars] : g_digRect[spares];
                q.subImage.imageRect.offset = { rc[0], rc[1] }; q.subImage.imageRect.extent = { rc[2], rc[3] };
                q.pose = IdentityPose();
                q.pose.position = { k == 0 ? -0.0175f : 0.0325f, -0.150f, -0.55f };
                q.size.width = k == 0 ? w : h * (float)kDigW / (float)kDigH; q.size.height = h;
                layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q);
            }
            static ULONGLONG s_batLog = 0; const ULONGLONG t = GetTickCount64();
            if (t - s_batLog > 10000) { s_batLog = t; XLOG("[OLVR][BATGAUGE] camcorder up: battery %.0f%% (%d bars), %d spare", energy * 100.0f, bars, spares); }
        }
    }

    // [POSEFREEZE] OLUE3_FrameTick is NOT called here any more - it ages GAME-frame counters
    // (split freshness, the head-injection ramp) and its own comment says "ticked once per
    // Present". Under the worker this ran at 90Hz against a ~35fps game, draining the split
    // freshness 2-3x too fast: the moment the game missed a split the mod declared stereo
    // stale and fell to the head-locked QUAD layer, then the next game frame revived it. That
    // flapping between a world-locked projection and a head-locked panel IS "flickering head
    // tracking" (visible in the log as layer=STEREO/quad alternating). It now ticks once per
    // Present in both modes, from OLXR_OnPresent.
    XrFrameEndInfo fei = {}; fei.type = XR_TYPE_FRAME_END_INFO_VALUE;
    fei.displayTime = fs.predictedDisplayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE_VALUE;
    fei.layerCount = layerCount; fei.layers = layerCount ? layers : nullptr;
    // [XRENDF] ME1's lesson, ported: a runtime rejecting the layers is indistinguishable from
    // success if the result is discarded (SteamVR validates and rejects; Meta forgives). The
    // result also rides the periodic log below, so a run answers this without a repro recipe.
    const double te = PerfNowMs();
    const XrResult efr = g_fn.endFrame(g_session, &fei);
    {
        const double de = PerfNowMs() - te;
        AcquireSRWLockExclusive(&g_pfLock); g_pfEnd.Add(de); ReleaseSRWLockExclusive(&g_pfLock);
    }
    g_lastEndFrame = (LONG)efr;
    if (!XrSucceeded(efr)) {
        static unsigned s_efLogs = 0;
        const unsigned n = s_efLogs++;
        if (n < 8 || (n % 300) == 0)
            XLOG("[OLVR][XRENDF] xrEndFrame FAILED r=%d layers=%d", (int)efr, layerCount);
    }

    ++g_frame;
    static DWORD s_lastLog = 0;
    DWORD now = NowMs();
    if (now - s_lastLog > 3000) {
        s_lastLog = now;
        // Log the per-eye FOV CONTINUOUSLY, not once at startup. Outlast's camera FOV moves
        // constantly during play - DefaultFOV=90, RunningFOV=100, and the camcorder zooms down
        // to CamcorderMinFOV=15 - and under AspectRatio_MaintainXFOV that value IS the horizontal
        // FOV each eye renders. A single early sample says nothing about the rest of the run;
        // one taken during a camcorder zoom reads ~32deg and looks like a bug that isn't one.
        float fx = 0.0f, fy = 0.0f;
        OLUE3_GetEyeFov(&fx, &fy);
        // bb/eye sizes and the headset's own FOV: the "menus and cutscenes do not fill the eye"
        // class of bug is decided by WHICH path the frame took (STEREO vs quad) and how the source
        // aspect compares to the lens. Guessing that from a description is how wrong builds happen.
        XLOG("[OLVR][XR] bb=%ux%u eye=%ux%u hmdFov=%.0fx%.0fdeg slack=%d",
             g_w, g_h, g_eyeW, g_eyeH, g_rtFovX * 57.2957795f, g_rtFovY * 57.2957795f, g_eyeSlack);
        // gameFrames is the published-frame counter: (delta between two log lines)/3s = the
        // game's real fps, while frames= is the submit cadence - under the worker the two
        // are DIFFERENT numbers, and their ratio is this whole fix working.
        // [HEADPOS] head=(x,y,z) is the position offset in METRES from the current baseline and
        // pos= says whether lean/duck is applying it. A seated human stays within ~0.2 m of
        // their own centre; a steady y of +0.8 or more with pos=armed is a desk baseline - recenter.
        XLOG("[OLVR][XR] frames=%lld layer=%s pose=%ld yaw=%.1f pitch=%.1f head=(%.2f,%.2f,%.2f)m pos=%s eyeFov=%.1fx%.1fdeg state=%s ef=%ld gameFrames=%llu",
             g_frame, sbs ? (g_screenOn ? "SCREEN" : "STEREO") : (g_olUseProjection ? "proj" : "quad"), (long)g_olPoseValid,
             g_olYawRad * 57.2957795f, g_olPitchRad * 57.2957795f,
             g_olHeadX, g_olHeadY, g_olHeadZ,
             !OLUE3_GetHeadPos() ? "off" : (g_posArmed ? "armed" : (g_posPickedUp ? "settling" : "waiting")),
             fx * 57.2957795f, fy * 57.2957795f,
             SessionStateName(g_sessionState), (long)g_lastEndFrame,
             (unsigned long long)(dev ? 0 : g_conSeq));
        // [XRSUBMIT] The full submitted truth, ME1's shape. Reads the doubling question
        // directly: rtEyeSep is the separation the RUNTIME expects between the two eye
        // viewpoints; this mod declares ONE pose for both (declSep is therefore 0 by construction).
        // If a runtime honours its own eye points, the submitted images carry this mod's own
        // separation AND its reprojection adds that gap on top - the over-separated, unfusable picture.
        if (g_rtEyeFovValid) {
            const float dx = g_rtEyePose[1].position.x - g_rtEyePose[0].position.x;
            const float dy = g_rtEyePose[1].position.y - g_rtEyePose[0].position.y;
            const float dz = g_rtEyePose[1].position.z - g_rtEyePose[0].position.z;
            XLOG("[OLVR][XRSUBMIT] rtEyeSep=%.4fm rtFov e0 %.1f/%.1f e1 %.1f/%.1f | decl e0 %.1f/%.1f "
                 "rect %dx%d+%d,%d | decl e1 %.1f/%.1f rect %dx%d+%d,%d | declPose q=(%.3f,%.3f,%.3f,%.3f)",
                 sqrtf(dx * dx + dy * dy + dz * dz),
                 g_rtEyeFov[0].angleLeft * 57.2957795f, g_rtEyeFov[0].angleRight * 57.2957795f,
                 g_rtEyeFov[1].angleLeft * 57.2957795f, g_rtEyeFov[1].angleRight * 57.2957795f,
                 g_dbgDeclFov[0].angleLeft * 57.2957795f, g_dbgDeclFov[0].angleRight * 57.2957795f,
                 g_dbgRect[0][2], g_dbgRect[0][3], g_dbgRect[0][0], g_dbgRect[0][1],
                 g_dbgDeclFov[1].angleLeft * 57.2957795f, g_dbgDeclFov[1].angleRight * 57.2957795f,
                 g_dbgRect[1][2], g_dbgRect[1][3], g_dbgRect[1][0], g_dbgRect[1][1],
                 g_dbgDeclPose.orientation.x, g_dbgDeclPose.orientation.y,
                 g_dbgDeclPose.orientation.z, g_dbgDeclPose.orientation.w);
        }
    }
    return true;
}

// [SUBMITTHREAD] The worker: a clean loop whose only blocking cost is waitFrame itself, so the
// compositor sees its cadence answered every single interval no matter what the game is doing.
static DWORD WINAPI WorkerMain(LPVOID) {
    while (!g_workerStop) {
        // [EXITGUARD] Step OUT of the OpenXR runtime as soon as the game stops presenting.
        // When the game closes, Windows tears the process down while this thread may be parked
        // inside xrWaitFrame; the runtime's own teardown then blocks on a lock this thread
        // holds and the process hangs forever with no window, appearing to still be running after
        // the game was closed. A thread sitting in Sleep() is safe to tear down, one inside the
        // runtime is not. Present going quiet is also exactly what a shutdown looks like from
        // here, and if it resumes - a long load, a hitch - playback simply picks back up.
        const DWORD quiet = NowMs() - g_lastPresentMs;
        if (g_lastPresentMs && quiet > 2000) { Sleep(50); continue; }
        PollEvents();
        if (g_dead) break;
        if (!g_running) { Sleep(10); continue; }
        ConsumeUpload();
        if (!RunXrFrame(nullptr, g_conSeq != 0)) Sleep(20);
    }
    return 0;
}

void OLXR_Shutdown() {
    // This runs from DllMain(PROCESS_DETACH), which holds the LOADER LOCK. Waiting on the
    // worker there is a guaranteed deadlock - the thread cannot complete its exit while this
    // lock is held, so the game "closes" and the process stays alive forever (measured
    // 2026-08-04, requiring the process to be killed manually). Signal and walk away: the process is being
    // torn down, and destroying XR objects underneath a thread that may still be inside an
    // OpenXR call is worse than leaving the OS to reclaim them.
    if (g_workerThread) {
        InterlockedExchange(&g_workerStop, 1);
        CloseHandle(g_workerThread); g_workerThread = nullptr;
        g_dead = true;              // stops the worker at its next loop check
        return;
    }
    if (g_session) { if (g_running) g_fn.endSession(g_session); g_fn.destroySession(g_session); g_session = 0; }
    if (g_sysSurf) { g_sysSurf->Release(); g_sysSurf = nullptr; }
    if (g_uploadTex) { g_uploadTex->Release(); g_uploadTex = nullptr; }
    if (g_blitTmpRtv) { g_blitTmpRtv->Release(); g_blitTmpRtv = nullptr; }
    if (g_blitTmp)    { g_blitTmp->Release();    g_blitTmp = nullptr; }
    if (g_blitBlend)  { g_blitBlend->Release();  g_blitBlend = nullptr; }
    if (g_blitDepth)  { g_blitDepth->Release();  g_blitDepth = nullptr; }
    if (g_blitRaster) { g_blitRaster->Release(); g_blitRaster = nullptr; }
    if (g_ctx) { g_ctx->Release(); g_ctx = nullptr; }
    if (g_d3d11) { g_d3d11->Release(); g_d3d11 = nullptr; }
}
