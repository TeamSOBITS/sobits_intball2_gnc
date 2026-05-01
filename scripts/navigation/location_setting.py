#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import rospy
import tf2_ros
import yaml
import rospkg # パッケージパス取得用
import tkinter as tk
from tkinter import messagebox
from subprocess import Popen, PIPE


def resolve_package_path(*relative_parts):
    """Resolve path under sobits_intball2_gnc using rospack with workspace fallback."""
    try:
        rospack = rospkg.RosPack()
        base_path = rospack.get_path('sobits_intball2_gnc')
    except Exception:
        base_path = os.path.expanduser("~/catkin_ws/src/sobits_intball2_gnc/")
    return os.path.join(base_path, *relative_parts)

class ISSLocationSetting(tk.Tk):
    def __init__(self):
        super(ISSLocationSetting, self).__init__()
        
        # ROS 1 初期化
        rospy.init_node('location_setting_node', anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 設定：ISS基準座標と機体中心
        self.target_frame = "body"      
        self.reference_frame = "iss_body" 
        
        # 保存先ディレクトリの確定 (sobits_intball2_gnc/maps)
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('sobits_intball2_gnc')
            self.default_dir = os.path.join(pkg_path, 'maps')
        except Exception:
            # フォールバック：通常のワークスペース構成を想定
            self.default_dir = os.path.expanduser("~/catkin_ws/src/sobits_intball2_gnc/maps")
        
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir, exist_ok=True)

        self.location_path = ""
        self.locations_data = {"location_pose": {}}

        # GUI初期設定
        self.title("ISS Location Setting")
        self.geometry("600x500")
        
        # ファイル選択 (起動時に実行)
        self.ready = False
        if not self.select_location_file():
            rospy.logerr("No file selected. Exiting.")
            self.destroy()
            return

        self.create_widgets()
        self.refresh_list()
        self.ready = True

    def select_location_file(self):
        """Zenityを使用して保存先YAMLを選択"""
        default_file = os.path.join(self.default_dir, "iss_locations.yaml")

        proc = Popen(["zenity", "--file-selection", "--save", "--confirm-overwrite", 
                      "--title=Select Location YAML", f"--filename={default_file}"],
                     stdout=PIPE, stderr=PIPE)
        out, _ = proc.communicate()
        path = out.decode('utf-8').strip()

        if not path:
            return False
        
        if not path.endswith(".yaml"):
            path += ".yaml"
        
        self.location_path = path
        self.load_yaml()
        return True

    def load_yaml(self):
        """YAMLの読み込み"""
        if os.path.exists(self.location_path):
            try:
                with open(self.location_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and "location_pose" in data:
                        self.locations_data = data
                    else:
                        self.locations_data = {"location_pose": {}}
            except Exception as e:
                rospy.logwarn(f"Failed to load yaml: {e}")
                self.locations_data = {"location_pose": {}}
        else:
            self.locations_data = {"location_pose": {}}

    def save_yaml(self):
        """YAMLの保存（アトミック書き込み）"""
        try:
            tmp_path = self.location_path + ".tmp"
            # 元ファイルのパーミッションを事前に保存
            orig_mode = None
            if os.path.exists(self.location_path):
                orig_mode = os.stat(self.location_path).st_mode
            with open(tmp_path, 'w') as f:
                yaml.dump(self.locations_data, f, default_flow_style=False)
            os.replace(tmp_path, self.location_path)
            # パーミッション復元
            if orig_mode is not None:
                os.chmod(self.location_path, orig_mode)
            rospy.loginfo(f"Saved to {self.location_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file: {e}")

    def create_widgets(self):
        """GUIレイアウト構築"""
        reg_frame = tk.LabelFrame(self, text="Register Current Position", padx=10, pady=10)
        reg_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(reg_frame, text="Location Name:").grid(row=0, column=0, sticky="w")
        self.name_entry = tk.Entry(reg_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5)
        
        btn_add = tk.Button(reg_frame, text="SNAP CURRENT POS", command=self.add_current_position, 
                           bg="#007BFF", fg="white", font=("", 10, "bold"))
        btn_add.grid(row=0, column=2, padx=5)

        list_frame = tk.LabelFrame(self, text=f"Registered Locations", padx=10, pady=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(list_frame)
        self.scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def refresh_list(self):
        """UIリストの更新"""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        poses = self.locations_data.get("location_pose", {})
        for name in sorted(poses.keys()):
            row = tk.Frame(self.scroll_frame, pady=2, bd=1, relief="groove")
            row.pack(fill="x", expand=True, padx=2, pady=2)

            tk.Label(row, text=name, width=20, anchor="w", font=("", 10, "bold")).pack(side="left", padx=5)
            
            p = poses[name]["translation"]
            coord_text = f"X:{p['x']:.2f} Y:{p['y']:.2f} Z:{p['z']:.2f}"
            tk.Label(row, text=coord_text, fg="#555").pack(side="left", padx=5)

            tk.Button(row, text="Delete", command=lambda n=name: self.delete_location(n), fg="red").pack(side="right", padx=5)
            tk.Button(row, text="Rename", command=lambda n=name: self.rename_location(n)).pack(side="right")

    def add_current_position(self):
        """現在のTFをキャプチャしてクォータニオンのみで保存"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Location name is empty.")
            return
        
        try:
            # iss_body座標系におけるbodyの最新位置を取得
            trans = self.tf_buffer.lookup_transform(self.reference_frame, self.target_frame, 
                                                   rospy.Time(0), rospy.Duration(1.0))
            
            t = trans.transform.translation
            q = trans.transform.rotation

            # クォータニオン一本に絞ったデータ構造
            new_entry = {
                "translation": {
                    "x": float(t.x),
                    "y": float(t.y),
                    "z": float(t.z)
                },
                "rotation": {
                    "x": float(q.x),
                    "y": float(q.y),
                    "z": float(q.z),
                    "w": float(q.w)
                }
            }

            self.locations_data["location_pose"][name] = new_entry
            self.save_yaml()
            self.refresh_list()
            self.name_entry.delete(0, tk.END)
            rospy.loginfo(f"Location '{name}' has been registered (Quaternion only).")

        except Exception as e:
            messagebox.showerror("TF Error", f"Failed to get transform: {e}")

    def delete_location(self, name):
        if messagebox.askyesno("Confirm", f"Delete '{name}'?"):
            del self.locations_data["location_pose"][name]
            self.save_yaml()
            self.refresh_list()

    def rename_location(self, old_name):
        from tkinter import simpledialog
        new_name = simpledialog.askstring("Rename", f"Enter new name for '{old_name}':")
        if new_name and new_name != old_name:
            if new_name in self.locations_data["location_pose"]:
                messagebox.showerror("Error", "That name already exists.")
                return
            self.locations_data["location_pose"][new_name] = self.locations_data["location_pose"].pop(old_name)
            self.save_yaml()
            self.refresh_list()

if __name__ == '__main__':
    try:
        app = ISSLocationSetting()
        if app.ready:
            app.mainloop()
    except rospy.ROSInterruptException:
        pass
