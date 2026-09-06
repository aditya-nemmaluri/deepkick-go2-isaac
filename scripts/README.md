cat << 'EOF' > scripts/README.md
# DeepKick Go2 Scripts Directory

This directory contains execution scripts for training, evaluating, visualizing, and rendering the Unitree Go2 ball-kicking policy.

---

## 📜 Script Reference

* **`train.py`**: Primary training entry point. Imports modular scene and MDP definitions from `config/env_cfg.py` and trains PPO across parallel environments via `rsl_rl`.
* **`train_standalone.py`**: Self-contained, single-file training script containing all scene definitions, reward terms, and PPO configs inline. Ideal for fast debugging without external module dependencies.
* **`eval.py`**: Headless physics benchmark. Loads a trained policy checkpoint (e.g., `checkpoints/model_1499.pt`) and outputs quantitative metrics like peak ball velocity, kick displacement, and success rates without requiring a display server.
* **`play.py`**: Real-time 3D interactive viewer for inspecting policy behavior in Isaac Sim (meant for execution with X11, VNC, or `DISPLAY=:99`).
* **`render_video.py`**: Offscreen camera renderer. Steps through policy inference using offscreen camera sensors and exports an MP4 video (`go2_kick_demo.mp4`) directly to disk.

---

## 🚀 Commands Quick Reference

Run all scripts from your `IsaacLab` directory using `./isaaclab.sh -p`:

```bash
# 1. Start Training (Headless, 4096 Envs)
./isaaclab.sh -p ~/deepkick-go2-isaac/scripts/train.py --num_envs 4096

# 2. Run Headless Physics Evaluation
./isaaclab.sh -p ~/deepkick-go2-isaac/scripts/eval.py

# 3. Render Offscreen MP4 Video Demo
DISPLAY=:99 ./isaaclab.sh -p ~/deepkick-go2-isaac/scripts/render_video.py

# 4. Interactive 3D Playback (noVNC / GUI)
DISPLAY=:99 ./isaaclab.sh -p ~/deepkick-go2-isaac/scripts/play.py --num_envs 16
```
