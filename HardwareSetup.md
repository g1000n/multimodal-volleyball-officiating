# Hardware Setup: Two-iPhone Rig (Camera + Microphone)

Setup instructions for the physical devices this system depends on:
**iPhone 14** running Camo (camera, for gesture recognition) and
**iPhone 13** running Camo, paired as a second device (microphone, for whistle detection).

This is a companion to the main `README.md` (which covers the software/data
pipeline) -- this file covers getting the two phones actually talking to
`live_deployment.py` correctly.

---

## What you need

| Role | Device | App (free) | Connects via |
|---|---|---|---|
| Camera | iPhone 14 | Camo Camera + Camo Studio | USB (recommended) or Wi-Fi |
| Microphone | iPhone 13 | Camo Camera + Camo Studio | USB (recommended) or Wi-Fi |

Both phones use the **same app** -- since Camo Studio 1.2, it supports pairing
two separate devices at once: one supplies the camera feed, a second can be
selected specifically as the audio source. This avoids introducing a second,
less-vetted app (an earlier draft of this doc recommended WO Mic; that's been
replaced after re-evaluating it for trust/safety -- see note below).

**Historical note, kept for context:** earlier testing hit a real bug where
`WHISTLE_DEVICE_INDEX` left at `None` caused the OS to silently pick the
wrong input (a Camo virtual mic channel) with no error, just total silence.
That was a genuine bug at the time, because Camo wasn't yet being used
*deliberately* for the mic. Now that both phones are intentionally routed
through Camo, seeing a Camo device in your input list is expected and
correct -- the thing to verify instead is that Camo's Audio Settings are
actually pointed at the iPhone 13, not defaulting to the iPhone 14.

---

## Part 1: Camera (iPhone 14 + Camo)

1. Install **Camo Camera** on the iPhone 14, and **Camo Studio** on the PC.
2. Connect the iPhone 14 to the PC via USB cable (lowest latency, most
   reliable). Open Camo on both ends.
3. Find which camera index Windows assigned to it:
   ```powershell
   python list_cameras.py
   ```
   This cycles through each available camera index and shows a preview
   window with the index in the title. Note whichever one shows the
   iPhone's feed.
4. Set that index in `live_deployment.py`:
   ```python
   CAMERA_INDEX = 1   # <- change to whatever list_cameras.py showed you
   ```

---

## Part 2: Microphone (iPhone 13 + Camo, as a second paired device)

Camo Studio can pair a second device specifically as an audio source,
independent of which device supplies the video. This is what we'll use for
the referee's mic, so no second app is needed.

1. Install **Camo Camera** on the iPhone 13 too (same free app as the
   iPhone 14 -- nothing new to install).
2. Connect the iPhone 13 to the PC via USB cable as well (both phones
   connected at the same time). Open Camo Camera on it.
3. In **Camo Studio** on the PC, it should now detect both connected
   devices. Confirm:
   - **Camera source** is still set to the iPhone 14
   - Open **Audio Settings** and explicitly select the **iPhone 13's
     microphone** as the audio source (this is the step that's easy to
     miss -- if you don't select it here, Camo may default to the iPhone
     14's own mic instead, which defeats the point of the second phone)
4. Camo now exposes **one combined virtual device** to Windows -- the
   iPhone 14's video merged with the iPhone 13's audio.
5. Find which input index that combined device registered as:
   ```powershell
   python list_mics.py
   ```
   This lists every audio input device, then actually records 2 seconds
   from each one and shows a peak-level bar.
6. **Verify it's actually the iPhone 13, not the iPhone 14:** blow a
   whistle or speak near the iPhone 13, then repeat near the iPhone 14.
   The Camo device's peak should respond to the iPhone 13 and stay quiet
   for the iPhone 14. If it's backwards, go back to Camo Studio's Audio
   Settings and re-check which device is selected -- don't proceed until
   this is confirmed, since a silent misconfiguration here is exactly the
   kind of failure that's easy to miss until game day.
7. Set that confirmed index in `live_deployment.py`:
   ```python
   WHISTLE_DEVICE_INDEX = None   # <- change to your confirmed-working index, e.g. 3
   ```
   **Do not leave this as `None`** in a real deployment -- that lets the OS
   pick a default, which is the original failure mode this whole setup is
   designed to avoid.

---

## Part 3: Test each device in isolation before combining them

Don't jump straight to the full `live_deployment.py` -- verify each half
works alone first, so if something's wrong you know which half to debug.

**Camera alone:** confirmed already via the `list_cameras.py` preview step.

**Microphone alone:**
```powershell
python whistle_detector.py 3
```
(replace `3` with your actual `WHISTLE_DEVICE_INDEX`). Blow a whistle near
the iPhone 13 and confirm detections print to the console. If nothing
triggers, re-check the index with `list_mics.py` -- and double-check Camo
Studio's Audio Settings are still pointed at the iPhone 13 -- before
assuming the model itself is at fault.

**Only once both check out independently:**
```powershell
python live_deployment.py
```

---

## Physical placement

- **Camera (iPhone 14):** opposite side of the court from the referee, per
  the thesis's stated camera design -- wider, more consistent view of arm
  movements while minimizing occlusion from the net/pole/players.
- **Microphone (iPhone 13):** attached close to the referee -- clipped to a
  shirt pocket, lanyard, or armband. Proximity matters more than app choice
  for whistle pickup quality; don't leave it propped somewhere across the
  court.

---

## Before a real game/demo day: recalibration check

`whistle_detector.py` has a constant, `GATE_MAX_PITCH_INSTABILITY`, that the
code explicitly flags as **mic-specific, not universal**:

> *"if you test on a DIFFERENT mic later (e.g. the real game-day phone
> setup), re-check this value against a fresh log rather than assuming it
> still fits."*

Switching from whatever mic was used during earlier development testing to
the iPhone 13 routed through Camo is exactly the scenario this warns about.
After your first real test session with this rig:

1. Check the whistle detection log (`data/whistle_logs/`, or wherever your
   session logging writes to) for the `pitch_instability` values real
   whistles produced through this specific mic.
2. If real whistles are landing at or above the current `20.0` threshold and
   getting blocked by the gate, raise it -- but only based on a fresh log
   from this exact hardware, not by assumption.

---

## Pre-flight checklist (repeat before every session)

Device indices can shift after a reboot or reconnect -- don't assume
yesterday's numbers still hold.

- [ ] iPhone 14 connected via USB, Camo running on both ends
- [ ] iPhone 13 connected via USB, Camo Camera running, selected as the audio source in Camo Studio's Audio Settings
- [ ] Re-run `list_cameras.py` -- confirm `CAMERA_INDEX` still matches
- [ ] Re-run `list_mics.py` -- confirm `WHISTLE_DEVICE_INDEX` still matches
- [ ] `python whistle_detector.py <index>` -- confirm live whistle detection
      works standalone
- [ ] Only then: `python live_deployment.py`

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Mic produces total silence, no errors | `WHISTLE_DEVICE_INDEX` is `None` or pointing at the wrong device -- re-run `list_mics.py` |
| Detected audio seems to be coming from the iPhone 14 instead of the iPhone 13 | Camo Studio's Audio Settings defaulted to the camera device's own mic -- re-open Audio Settings and explicitly re-select the iPhone 13 |
| Console says "whistle_detector.py not found -- MANUAL WHISTLE MODE" | `models/whistle_svm_model.pkl` isn't present or `whistle_detector.py` isn't importable from the working directory you launched from |
| Real whistles aren't triggering during actual gameplay (but work standing still near the mic) | Expected per `live_deployment.py`'s own comments -- detection was validated close to the mic, standing still; movement/distance/ambient noise during real play may reduce reliability. This is exactly why `REQUIRE_WHISTLE_FOR_SCORING` defaults to `False` |
| A single whistle blast reports as 2-3 separate detections | `WINDOW_SEC` in `whistle_detector.py`/`08_realtime_ui.py` doesn't match `TARGET_LEN` in `02_extract_whistle_clips.py` -- see the main `README.md`'s note on this |
| Camera index or mic index "worked yesterday" but not today | Windows reassigned indices after reboot/reconnect -- re-run `list_cameras.py` / `list_mics.py`, don't trust old numbers |