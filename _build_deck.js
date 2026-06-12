const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
p.author = "PHYSWM";
p.title = "Embedding physics into a world model learned from pixels";

const W = 13.33, H = 7.5;
const FIG = "C:/Users/donov/OneDrive/Desktop/PHYSWM/results/_deck/";
const RES = "C:/Users/donov/OneDrive/Desktop/PHYSWM/results/";

// palette
const NAVY = "16243F", BLUE = "2A7FB8", TEAL = "1C7293", AMBER = "E0922F";
const GRAY = "9AA3AD", BG = "F7F9FB", INK = "1E293B", MUTED = "64748B";
const WHITE = "FFFFFF", GOOD = "2E7D5B", BAD = "C0504D", CARD = "FFFFFF";
const HEAD = "Trebuchet MS", BODY = "Calibri", MONO = "Consolas";

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.12 });

function header(s, num, title) {
  s.background = { color: BG };
  s.addShape(p.shapes.OVAL, { x: 0.55, y: 0.42, w: 0.62, h: 0.62, fill: { color: BLUE } });
  s.addText(String(num), { x: 0.55, y: 0.42, w: 0.62, h: 0.62, align: "center", valign: "middle",
    color: WHITE, fontFace: HEAD, fontSize: 24, bold: true });
  s.addText(title, { x: 1.35, y: 0.4, w: 11.6, h: 0.7, align: "left", valign: "middle",
    color: INK, fontFace: HEAD, fontSize: 28, bold: true, margin: 0 });
}
function footer(s, n) {
  s.addText("Embedding physics into a world model learned from pixels  |  2-week update",
    { x: 0.55, y: 7.05, w: 10, h: 0.3, color: MUTED, fontFace: BODY, fontSize: 9, margin: 0 });
  s.addText(String(n), { x: 12.5, y: 7.05, w: 0.5, h: 0.3, color: MUTED, fontFace: BODY, fontSize: 9, align: "right" });
}

// ---------------- 1. Title ----------------
let s = p.addSlide();
s.background = { color: NAVY };
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: AMBER } });
s.addText("SUMMER RESEARCH  •  TWO-WEEK UPDATE", { x: 1.0, y: 1.15, w: 11, h: 0.4, color: AMBER,
  fontFace: HEAD, fontSize: 15, bold: true, charSpacing: 3, margin: 0 });
s.addText("Embedding known physics into a\nworld model learned from pixels", { x: 0.95, y: 1.6, w: 11.6, h: 1.9,
  color: WHITE, fontFace: HEAD, fontSize: 40, bold: true, margin: 0, lineSpacingMultiple: 1.02 });
s.addText("A unicycle robot, a JEPA, and what a kinematic prior actually buys",
  { x: 1.0, y: 3.55, w: 11.5, h: 0.5, color: "CADCFC", fontFace: BODY, fontSize: 19, italic: true, margin: 0 });
s.addText([
  { text: "Punchline:  ", options: { bold: true, color: WHITE } },
  { text: "embedding physics helps the model recover true state, but only when that state is ", options: { color: "CADCFC" } },
  { text: "visible in the pixels", options: { color: AMBER, bold: true } },
  { text: ", and a simple change to the ", options: { color: "CADCFC" } },
  { text: "task", options: { color: AMBER, bold: true } },
  { text: " can match it. Net benefit so far: real but narrow.", options: { color: "CADCFC" } },
], { x: 1.0, y: 4.55, w: 11.3, h: 1.0, fontFace: BODY, fontSize: 16, margin: 0 });
s.addText("[Your name]   •   Advisor: [Advisor]   •   June 2026",
  { x: 1.0, y: 6.5, w: 11, h: 0.4, color: "8FA3C8", fontFace: BODY, fontSize: 13, margin: 0 });
s.addNotes("Frame the whole talk as my summer goal: how to embed known physics into a model learned from camera feed. The heading-flip story is the testbed for that question. Give the punchline up front: physics helps, but narrowly and conditionally.");

// ---------------- 2. Goal + testbed ----------------
s = p.addSlide();
header(s, 1, "The goal, and a testbed to study it");
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.3, w: 12.2, h: 1.35, fill: { color: "EEF3F8" }, line: { color: "C9D8E6", width: 1 } });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.3, w: 0.12, h: 1.35, fill: { color: TEAL } });
s.addText([
  { text: "The research question:  ", options: { bold: true, color: TEAL } },
  { text: "if we know some physics for an engineering system but want to build a dynamics model from a camera feed, how should we embed that knowledge into the learned model?", options: { italic: true, color: INK } },
], { x: 0.85, y: 1.4, w: 11.7, h: 1.15, fontFace: BODY, fontSize: 16, valign: "middle", margin: 0 });
s.addImage({ path: FIG + "fig0_world.png", x: 0.9, y: 2.95, w: 11.5, h: 2.5, sizing: { type: "contain", w: 11.5, h: 2.5 } });
s.addText([
  { text: "Testbed: ", options: { bold: true, color: INK } },
  { text: "a triangle robot whose physics we know exactly (unicycle kinematics: x' = v cos θ, y' = v sin θ, θ' = ω), observed only as 64x64 pixels. ", options: { color: INK } },
  { text: "Can the model recover the true state (x, y, θ) from pixels, and does embedding the kinematics help?", options: { bold: true, color: TEAL } },
], { x: 0.7, y: 5.7, w: 12.0, h: 1.0, fontFace: BODY, fontSize: 14.5, align: "center", margin: 0 });
footer(s, 2);
s.addNotes("State the summer goal verbatim, then show the testbed is a clean instance of it: known physics (unicycle), hidden behind pixels. The whole point is that I control the physics exactly, so I can measure what embedding it does. Mention the two-representation rule: model sees pixels only; (x,y,theta) is held out as an answer key.");

// ---------------- 3. Three predictors ----------------
s = p.addSlide();
header(s, 2, "The model: a JEPA with three predictor variants");
s.addText([
  { text: "JEPA: ", options: { bold: true, color: INK } },
  { text: "a CNN encoder maps a frame to a latent z; a predictor maps (z, action) to the next z; trained to match the encoder's output on the next frame. We embed physics by changing the predictor. Three variants, increasing prior knowledge left to right:", options: { color: MUTED } },
], { x: 0.55, y: 1.2, w: 12.2, h: 0.7, fontFace: BODY, fontSize: 14, margin: 0 });
const pm = [
  { x: 0.55, tag: "MLP", eq: "next_z = f(z, a)", prior: "No prior", desc: "Predicts the whole next latent from scratch. Maximum flexibility, zero structure. In practice it underfit: worse than just copying z.", accent: GRAY },
  { x: 4.83, tag: "RESIDUAL", eq: "next_z = z + f(z, a)", prior: "Generic prior", desc: "Predicts only the change, and defaults to 'no change' at init. Encodes that state moves slowly. Our strong baseline.", accent: BLUE },
  { x: 9.11, tag: "PHYSICS", eq: "next_z = kin(z, a) + f(z, a)", prior: "Known-physics prior", desc: "Base step IS the unicycle kinematics on dims 0,1,2 read as (x, y, theta), with learnable unit scales; MLP corrects the rest.", accent: TEAL },
];
pm.forEach(c => {
  s.addShape(p.shapes.RECTANGLE, { x: c.x, y: 2.05, w: 3.67, h: 4.4, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
  s.addShape(p.shapes.RECTANGLE, { x: c.x, y: 2.05, w: 3.67, h: 0.7, fill: { color: c.accent } });
  s.addText(c.tag, { x: c.x, y: 2.05, w: 3.67, h: 0.7, color: WHITE, fontFace: HEAD, fontSize: 17, bold: true, align: "center", valign: "middle", margin: 0 });
  s.addShape(p.shapes.RECTANGLE, { x: c.x + 0.25, y: 2.95, w: 3.17, h: 0.62, fill: { color: "F1F5F9" } });
  s.addText(c.eq, { x: c.x + 0.25, y: 2.95, w: 3.17, h: 0.62, color: INK, fontFace: MONO, fontSize: 12.5, align: "center", valign: "middle", margin: 0 });
  s.addText(c.prior, { x: c.x + 0.25, y: 3.75, w: 3.17, h: 0.4, color: c.accent, fontFace: HEAD, fontSize: 14, bold: true, margin: 0 });
  s.addText(c.desc, { x: c.x + 0.25, y: 4.2, w: 3.17, h: 2.1, color: MUTED, fontFace: BODY, fontSize: 12.5, margin: 0 });
});
footer(s, 3);
s.addNotes("This is the spine of the research question made concrete. MLP = no prior (and it underfit, which is why we moved off it). Residual = a generic prior, 'states change slowly', our baseline. Physics = the known unicycle equations baked into the predictor. The ladder from no prior to known-physics prior is exactly what 'how much physics to embed' means. Stress that residual is the fair baseline to compare physics against, not MLP.");

// ---------------- 4. Position vs heading ----------------
s = p.addSlide();
header(s, 3, "Position is free. Heading is the hard case.");
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.3, w: 5.9, h: 2.5, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addText("POSITION", { x: 0.85, y: 1.5, w: 5.3, h: 0.4, color: GOOD, fontFace: HEAD, fontSize: 15, bold: true, charSpacing: 2, margin: 0 });
s.addText("R² = 0.99", { x: 0.85, y: 1.9, w: 5.3, h: 0.95, color: GOOD, fontFace: HEAD, fontSize: 48, bold: true, margin: 0 });
s.addText("recovered almost perfectly, every variant", { x: 0.85, y: 2.95, w: 5.3, h: 0.6, color: MUTED, fontFace: BODY, fontSize: 13.5, margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 6.85, y: 1.3, w: 5.9, h: 2.5, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addText("HEADING", { x: 7.15, y: 1.5, w: 5.3, h: 0.4, color: BAD, fontFace: HEAD, fontSize: 15, bold: true, charSpacing: 2, margin: 0 });
s.addText("28% flipped", { x: 7.15, y: 1.9, w: 5.5, h: 0.95, color: BAD, fontFace: HEAD, fontSize: 44, bold: true, margin: 0 });
s.addText("pointed backwards on ~1 in 4 frames", { x: 7.15, y: 2.95, w: 5.3, h: 0.6, color: MUTED, fontFace: BODY, fontSize: 13.5, margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 4.05, w: 12.2, h: 2.55, fill: { color: "EEF3F8" }, line: { color: "D6E2ED", width: 1 } });
s.addText("Heading is the natural test for a physics prior", { x: 0.85, y: 4.25, w: 11.6, h: 0.4, color: INK, fontFace: HEAD, fontSize: 16, bold: true, margin: 0 });
s.addText([
  { text: "Metric I track all talk: ", options: { bold: true, color: INK } },
  { text: "flip rate = share of frames the probe gets heading > 90 degrees wrong (chance 50%, solved near 0%). The triangle's nose is 1-2 px, so a heading and its opposite render almost the same frame (front/back aliasing).", options: { color: INK }, breakLine: true },
  { text: "Why heading and not position: the kinematics couple heading to motion (v cos θ). If embedding the physics helps anywhere, it should help here.", options: { color: TEAL, bold: true } },
], { x: 0.85, y: 4.7, w: 11.7, h: 1.7, fontFace: BODY, fontSize: 13.5, margin: 0, paraSpaceAfter: 7 });
footer(s, 4);
s.addNotes("Position is solved by everything, so it cannot tell variants apart. Heading is the discriminator, and it is exactly where the physics prior should bite because the kinematics tie heading to where the robot goes. Define flip rate clearly, it is the through-line number.");

// ---------------- 5. Attempt 1: physics direct ----------------
s = p.addSlide();
header(s, 4, "Attempt 1: embed the physics prior directly");
s.addText("On the raw binary pixels, nothing I tried moved heading. Flip rate stayed pinned near 28%:",
  { x: 0.55, y: 1.25, w: 12, h: 0.4, color: INK, fontFace: BODY, fontSize: 15, margin: 0 });
const rows = [
  [{ text: "What I varied", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
   { text: "Heading flip rate", options: { bold: true, color: WHITE, fill: { color: TEAL }, align: "center" } }],
  ["Resolution: 40 -> 64 -> 128 px", { text: "~27%   (no change)", options: { align: "center" } }],
  ["Predictor: MLP / residual / physics", { text: "~28%   (no change)", options: { align: "center" } }],
  ["Physics-consistency loss weight: 0.1 -> 100", { text: "~28%   (weight 100 collapsed all)", options: { align: "center" } }],
  ["Hard pose lock (MLP forbidden on x,y,theta)", { text: "~29%", options: { align: "center" } }],
];
s.addTable(rows, { x: 0.55, y: 1.8, w: 9.2, colW: [6.0, 3.2], rowH: 0.62,
  fontFace: BODY, fontSize: 14, color: INK, valign: "middle",
  border: { type: "solid", pt: 1, color: "D6E2ED" }, fill: { color: "FFFFFF" }, align: "left" });
s.addShape(p.shapes.RECTANGLE, { x: 10.0, y: 1.8, w: 2.78, h: 3.1, fill: { color: NAVY } });
s.addText("Embedding the kinematic prior changed heading by about zero. The strongest setting even collapsed the whole representation.",
  { x: 10.25, y: 1.8, w: 2.3, h: 3.1, color: WHITE, fontFace: BODY, fontSize: 14.5, bold: true, valign: "middle", margin: 0 });
s.addText([
  { text: "PCA aside: ", options: { bold: true, color: INK } },
  { text: "SIGReg (the anti-collapse term) pushes the latent toward isotropic, so heading sits on no single axis. The information is recoverable nonlinearly but not concentrated, which already hints the physics prior is not anchoring its dims.", options: { color: MUTED } },
], { x: 0.55, y: 5.15, w: 12.2, h: 1.5, fontFace: BODY, fontSize: 13.5, margin: 0 });
footer(s, 5);
s.addNotes("This is the honest negative result, and it is the interesting part for the physics question. I tried the physics prior at every strength (soft base, a consistency loss swept 0.1 to 100, and a hard architectural lock) and none of it helped heading on raw pixels. Set up the next slide: WHY did embedding physics do nothing.");

// ---------------- 6. Why physics failed ----------------
s = p.addSlide();
header(s, 5, "Why embedding physics alone failed");
function diagCard(x, tag, title, body) {
  s.addShape(p.shapes.RECTANGLE, { x, y: 1.4, w: 5.95, h: 3.45, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
  s.addShape(p.shapes.RECTANGLE, { x, y: 1.4, w: 5.95, h: 0.8, fill: { color: NAVY } });
  s.addText(tag, { x: x + 0.3, y: 1.4, w: 5.4, h: 0.8, color: AMBER, fontFace: HEAD, fontSize: 17, bold: true, valign: "middle", margin: 0 });
  s.addText(title, { x: x + 0.3, y: 2.4, w: 5.4, h: 0.5, color: INK, fontFace: HEAD, fontSize: 16, bold: true, margin: 0 });
  s.addText(body, { x: x + 0.3, y: 2.95, w: 5.4, h: 1.8, color: MUTED, fontFace: BODY, fontSize: 13.5, margin: 0 });
}
diagCard(0.55, "REASON 1", "The referenced state was not observable",
  "The prior reads heading off the latent, but the aliased triangle hides heading in the pixels. A physics prior cannot inject a quantity the encoder cannot perceive in the first place.");
diagCard(6.85, "REASON 2", "The soft prior was silently switched off",
  "The learnable unit scale a_pos drifted from 1.0 down to ~0.4, shrinking the kinematic step until it barely constrained anything. The MLP can also cancel the base. And the prior never claimed dims 0,1,2: pose stayed smeared across the latent.");
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 5.2, w: 12.23, h: 1.1, fill: { color: AMBER } });
s.addText("Lesson: embed physics on an OBSERVABLE state, and stop the model from down-weighting the prior.",
  { x: 0.55, y: 5.2, w: 12.23, h: 1.1, color: NAVY, fontFace: HEAD, fontSize: 18, bold: true, align: "center", valign: "middle", margin: 0 });
footer(s, 6);
s.addNotes("These two reasons are the core research insight. One: a physics prior is not a magic wand, it cannot create information about a state that is not in the input. Two: a soft prior with learnable scales gives the optimizer a way to ignore it, and we saw exactly that (a_pos shrank). Both are general lessons for embedding physics, not quirks of this toy.");

// ---------------- 7. The fix ----------------
s = p.addSlide();
header(s, 6, "The fix: make heading visible, and make the task need it");
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.45, w: 5.4, h: 2.4, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.45, w: 0.1, h: 2.4, fill: { color: TEAL } });
s.addText("Make it VISIBLE", { x: 0.85, y: 1.6, w: 5, h: 0.45, color: TEAL, fontFace: HEAD, fontSize: 17, bold: true, margin: 0 });
s.addText("Paint a bright nose dot on the robot: gray body, white front. Heading is now unambiguous in the pixels, so the state the physics references is finally observable.",
  { x: 0.85, y: 2.1, w: 5, h: 1.6, color: MUTED, fontFace: BODY, fontSize: 14, margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 4.1, w: 5.4, h: 2.4, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 4.1, w: 0.1, h: 2.4, fill: { color: BLUE } });
s.addText("Make it DEMANDED", { x: 0.85, y: 4.25, w: 5, h: 0.45, color: BLUE, fontFace: HEAD, fontSize: 17, bold: true, margin: 0 });
s.addText("Predict several steps ahead so heading actually determines the outcome. The physics prior is ONE way to make the task need heading; plain multi-step is another.",
  { x: 0.85, y: 4.75, w: 5, h: 1.6, color: MUTED, fontFace: BODY, fontSize: 14, margin: 0 });
s.addImage({ path: RES + "_marker_preview.png", x: 6.2, y: 1.7, w: 6.7, h: 4.4, sizing: { type: "contain", w: 6.7, h: 4.4 } });
s.addText("binary triangle (top) vs nose-dot marker (middle, bottom) across headings",
  { x: 6.2, y: 6.05, w: 6.7, h: 0.35, color: MUTED, fontFace: BODY, fontSize: 11, italic: true, align: "center", margin: 0 });
footer(s, 7);
s.addNotes("Two knobs, one per failure reason. The nose dot fixes observability (a one-channel render change). Multi-step fixes demand (a one-line data change). Note explicitly that the physics prior and multi-step are two different ways to make the task demand heading, which sets up the physics comparison later.");

// ---------------- 8. Histogram result ----------------
s = p.addSlide();
header(s, 7, "With the fix, heading is recovered");
s.addImage({ path: FIG + "fig3_heading_histogram.png", x: 0.5, y: 1.35, w: 8.7, h: 5.0, sizing: { type: "contain", w: 8.7, h: 5.0 } });
s.addShape(p.shapes.RECTANGLE, { x: 9.5, y: 1.7, w: 3.3, h: 2.1, fill: { color: NAVY } });
s.addText("28%  ->  3%", { x: 9.5, y: 1.92, w: 3.3, h: 0.95, color: AMBER, fontFace: HEAD, fontSize: 32, bold: true, align: "center", margin: 0 });
s.addText("of frames flipped backwards", { x: 9.5, y: 2.85, w: 3.3, h: 0.8, color: WHITE, fontFace: BODY, fontSize: 14, align: "center", margin: 0 });
s.addText([
  { text: "The gray binary run has a clear second hump near 180 degrees: the front/back flips.", options: { bullet: true, breakLine: true, color: INK } },
  { text: "With heading visible and demanded, almost no mass sits past the 90-degree flip line.", options: { bullet: true, breakLine: true, color: INK } },
  { text: "Median error also improves: 23 -> 15 degrees.", options: { bullet: true, color: INK } },
], { x: 9.5, y: 4.05, w: 3.3, h: 2.3, fontFace: BODY, fontSize: 13, margin: 0, paraSpaceAfter: 8 });
footer(s, 8);
s.addNotes("The payoff that the problem is solvable at all. The histogram makes the flip metric literal: the tail past 90 degrees is the front/back confusion and it nearly vanishes. This particular run is residual plus nose dot at two-step horizon, no physics, which sets up the honest physics comparison next.");

// ---------------- 9. Interaction ----------------
s = p.addSlide();
header(s, 8, "It is an interaction: you need BOTH");
s.addImage({ path: FIG + "fig1_interaction.png", x: 0.5, y: 1.4, w: 8.4, h: 5.0, sizing: { type: "contain", w: 8.4, h: 5.0 } });
s.addText([
  { text: "The 2x2 that proves the point.", options: { bold: true, breakLine: true, color: INK, fontSize: 16 } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Three of the four conditions stay near 25%.", options: { bullet: true, breakLine: true, color: INK } },
  { text: "Only the both-on cell (visible AND demanded) drops to 4%.", options: { bullet: true, breakLine: true, color: INK } },
  { text: "Nose dot alone does nothing; multi-step on the aliased binary does nothing.", options: { bullet: true, breakLine: true, color: INK } },
  { text: "Neither knob is sufficient by itself. That is the result.", options: { bullet: true, color: INK } },
], { x: 9.1, y: 1.7, w: 3.7, h: 4.6, fontFace: BODY, fontSize: 14, margin: 0, paraSpaceAfter: 8 });
footer(s, 9);
s.addNotes("An interaction is more convincing than a single improvement: it shows the mechanism. Walk left to right: same single-step bars, then the multi-step pair where only the visible bar falls off a cliff. This is the general principle, observable AND demanded, stated as data.");

// ---------------- 10. Where physics helps ----------------
s = p.addSlide();
header(s, 9, "So does embedding physics help? Yes, but narrowly.");
s.addImage({ path: FIG + "fig4_physics_vs_nophysics.png", x: 0.5, y: 1.4, w: 8.5, h: 5.0, sizing: { type: "contain", w: 8.5, h: 5.0 } });
s.addText([
  { text: "Physics substitutes for task demand.", options: { bold: true, breakLine: true, color: INK, fontSize: 16 } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "With heading visible, the physics prior cracks heading at a SINGLE step (10%) where the plain model fails (26%).", options: { bullet: true, breakLine: true, color: TEAL, bold: true } },
  { text: "But predict 2+ steps and the plain model matches or beats it (3% vs 9%).", options: { bullet: true, breakLine: true, color: INK } },
  { text: "So physics is a shortcut to the same answer that longer horizons reach on their own. Useful, but not a unique capability here.", options: { bullet: true, color: INK } },
], { x: 9.2, y: 1.7, w: 3.6, h: 4.6, fontFace: BODY, fontSize: 13.5, margin: 0, paraSpaceAfter: 8 });
footer(s, 10);
s.addNotes("This is the central physics result. Be precise: physics helps most when task demand is weak (single step). It buys you heading from a short horizon the plain model cannot use. Once you predict farther, the plain residual model catches up and at two steps even beats physics. So the marginal value of the prior shrinks as the task itself demands the state.");

// ---------------- 11. Benefits and drawbacks ----------------
s = p.addSlide();
header(s, 10, "Embedding physics: benefits and drawbacks");
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 5.95, h: 5.0, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addShape(p.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 5.95, h: 0.7, fill: { color: GOOD } });
s.addText("BENEFITS", { x: 0.85, y: 1.4, w: 5.4, h: 0.7, color: WHITE, fontFace: HEAD, fontSize: 16, bold: true, valign: "middle", margin: 0 });
s.addText([
  { text: "Horizon efficiency: learns heading from single-step prediction, where the plain model needs longer horizons.", options: { bullet: true, breakLine: true } },
  { text: "Uses only the latent and action: no ground-truth state leaks in.", options: { bullet: true, breakLine: true } },
  { text: "Gives an interpretable target: dims meant to be (x, y, theta).", options: { bullet: true, breakLine: true } },
  { text: "Held position steady where a plain multi-step model sometimes collapsed it.", options: { bullet: true } },
], { x: 0.85, y: 2.25, w: 5.4, h: 4.0, color: INK, fontFace: BODY, fontSize: 13.5, margin: 0, paraSpaceAfter: 9 });
s.addShape(p.shapes.RECTANGLE, { x: 6.85, y: 1.4, w: 5.95, h: 5.0, fill: { color: CARD }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
s.addShape(p.shapes.RECTANGLE, { x: 6.85, y: 1.4, w: 5.95, h: 0.7, fill: { color: BAD } });
s.addText("DRAWBACKS", { x: 7.15, y: 1.4, w: 5.4, h: 0.7, color: WHITE, fontFace: HEAD, fontSize: 16, bold: true, valign: "middle", margin: 0 });
s.addText([
  { text: "Cannot create unobservable information: useless if the state is not in the pixels.", options: { bullet: true, breakLine: true } },
  { text: "Bypassable: learnable scales shrink and the MLP can cancel the kinematic base.", options: { bullet: true, breakLine: true } },
  { text: "Did not claim its dims: pose stayed smeared, so the interpretability hope did not hold.", options: { bullet: true, breakLine: true } },
  { text: "Redundant once the task demands the state (plain multi-step matched it).", options: { bullet: true, breakLine: true } },
  { text: "Fragile when over-forced: loss weight 100 collapsed; hard lock was unstable at higher res.", options: { bullet: true } },
], { x: 7.15, y: 2.25, w: 5.4, h: 4.0, color: INK, fontFace: BODY, fontSize: 13.5, margin: 0, paraSpaceAfter: 8 });
footer(s, 11);
s.addNotes("The honest scorecard, which is what the mentor most wants. Benefits are real: short-horizon efficiency and stability. Drawbacks are the meat of the next phase: a soft prior is bypassable, it did not anchor its dims, and it is redundant once the task already demands the state. Do not oversell; the point is a clear-eyed read on when embedding physics is worth it.");

// ---------------- 12. Next steps ----------------
s = p.addSlide();
s.background = { color: NAVY };
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: AMBER } });
s.addText("Next steps: making the physics embedding stronger", { x: 0.95, y: 0.5, w: 11.8, h: 0.8, color: WHITE, fontFace: HEAD, fontSize: 30, bold: true, margin: 0 });
s.addText("WHERE THINGS STAND", { x: 0.95, y: 1.45, w: 11, h: 0.35, color: AMBER, fontFace: HEAD, fontSize: 13, bold: true, charSpacing: 2, margin: 0 });
s.addText([
  { text: "Embedding a kinematic prior helps recover true state only when that state is visible AND the task needs it.", options: { bullet: true, breakLine: true } },
  { text: "When it helps, it substitutes for longer prediction horizons. Net benefit so far: real but narrow.", options: { bullet: true } },
], { x: 0.95, y: 1.85, w: 11.7, h: 1.3, color: "E6ECF6", fontFace: BODY, fontSize: 14.5, margin: 0, paraSpaceAfter: 8 });
s.addText("HOW TO MAKE PHYSICS STRONGER", { x: 0.95, y: 3.25, w: 11, h: 0.35, color: AMBER, fontFace: HEAD, fontSize: 13, bold: true, charSpacing: 2, margin: 0 });
s.addText([
  { text: "Stop the bypass: calibrate or constrain the unit scales, and tie the pose dims to physical units.", options: { bullet: true, breakLine: true } },
  { text: "Force the prior to own its dims: a light decoder or anchor so dims 0,1,2 truly carry (x, y, theta).", options: { bullet: true, breakLine: true } },
  { text: "Measure where physics SHOULD win: extrapolation to unseen speeds, low-data efficiency, long-horizon rollout stability (not just in-distribution probe accuracy).", options: { bullet: true, breakLine: true } },
  { text: "Scale to harder physics: more coupled or constrained dynamics, where data alone struggles and the prior earns its keep.", options: { bullet: true, breakLine: true } },
  { text: "Confirm the multi-step result at 128 px (re-running; first attempt was unstable).", options: { bullet: true } },
], { x: 0.95, y: 3.65, w: 11.8, h: 3.2, color: "E6ECF6", fontFace: BODY, fontSize: 14.5, margin: 0, paraSpaceAfter: 7 });
s.addNotes("Close on the research arc, not just this toy. The two-line status is the takeaway. Then the forward plan, ordered: first fix the bypass and make the prior anchor its dims, because that is what stopped it from working. Then, crucially, measure physics where it is SUPPOSED to help (extrapolation, low data, long rollouts), since in-distribution probe accuracy is not where priors usually pay off. Then scale to harder dynamics. Invite the mentor to steer on which engineering system to target next.");

p.writeFile({ fileName: RES + "_deck/PHYSWM_research_update_physics.pptx" }).then(f => console.log("WROTE " + f));
