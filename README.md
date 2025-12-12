# step-by-step **install and build guide** for your friend — assuming they already have **ROS 2 Humble installed** on **Ubuntu 22.04**.


## clone my repository first


```bash
git clone https://github.com/future-html/ros-turtlebot3
```

## !if found error permission denied and has granted access by ed25519 key ==> try to fix clone by add rsa-4096 ssh keygen and add to github account
---

### 🛠️ Installation & Build Instructions

> ✅ **Prerequisite**: ROS 2 Humble is already installed.  
> 💡 Run all commands in a terminal.

---

#### 1. **Install Required System Packages**
```bash
# Install ROS 2 camera driver and OpenCV (DO NOT use pip for these!)
sudo apt update
sudo apt install -y ros-humble-usb-cam python3-opencv
```

> ⚠️ **Important**: Do **not** run `pip install opencv-python` — it will cause crashes!

---

#### 2. **(Optional) Install YOLO for Human Detection**
```bash
# Only if you want YOLO (recommended for better accuracy)
pip3 install opencv-python-headless ultralytics
```

> 🔸 If you skip this, the node will run in **color-only mode** (still works!).

---

#### 3. **Clone Your Repository**
```bash
git clone https://github.com/yourusername/ros2_turtlecv.git
cd ros2_turtlecv
```

> 📝 Replace `yourusername` with your actual GitHub username.

---

#### 4. **Build the Package**
```bash
colcon build --symlink-install
```

> 💡 This may take 1–2 minutes. Ignore warnings about missing `yolov8n.pt` — it downloads on first run.

---

#### 5. **Source the Workspace**
```bash
source install/setup.bash
```

> 🔁 Run this **every time** you open a new terminal to use your package.

---

#### 6. **Run the Node**
```bash
ros2 run turtlecv opencv
```

> 🎯 That’s it! The node will wait for a camera feed on `/camera/image_raw`.

---

### 📌 How to Use It

#### 📷 With a Real Camera:
In **Terminal 1**:
```bash
ros2 run usb_cam usb_cam_node_exe --ros-args -r __ns:=/camera
```

In **Terminal 2** (after `source install/setup.bash`):
```bash
ros2 run turtlecv opencv
```

#### 🖥️ With Simulated Camera (no hardware needed):
In **Terminal 1**:
```bash
ros2 run image_tools cam2image --ros-args -r image:=/camera/image_raw --param burger_mode:=true
```

In **Terminal 2**:
```bash
ros2 run turtlecv opencv
```

---

### ❓ Troubleshooting Tips

- **"Command not found: colcon"** → Install with: `sudo apt install python3-colcon-common-extensions`
- **"No module named 'cv_bridge'"** → You’re missing ROS 2 setup; run: `source /opt/ros/humble/setup.bash`
- **Bus error / crash** → You installed `opencv-python` via pip. Fix:
  ```bash
  pip3 uninstall opencv-python
  sudo apt install python3-opencv ros-humble-cv-bridge
  ```

---

### ✅ Summary of Commands (Copy-Paste Friendly)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y ros-humble-usb-cam python3-opencv

# 2. (Optional) Install YOLO and opencv
pip install opencv-python-headless ultralytics

# 3. Clone and build
git clone https://github.com/yourusername/ros2_turtlecv.git
cd ros2_turtlecv
colcon build --symlink-install
source install/setup.bash

# 4. Run
ros2 run turtlecv opencv
```
