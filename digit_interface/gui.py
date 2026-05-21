import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import datetime
import os

# Import the new Windows-compatible Digit class
try:
    from .digit_windows import Digit
except ImportError:
    from digit_windows import Digit


class DigitInterfaceGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Internal state
        self.digit = None
        self.current_frame = None
        
        # --- UI LAYOUT ---
        self.main_container = tk.Frame(window)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL: Video frame
        self.video_frame = tk.Frame(self.main_container)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, width=480, height=640, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack(pady=10)

        # RIGHT PANEL: Settings and Controls
        self.control_frame = tk.Frame(self.main_container, width=250, bd=2, relief=tk.GROOVE)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        tk.Label(self.control_frame, text="DIGIT Settings", font=("Arial", 14, "bold")).pack(pady=(10, 20))

        # 1. Camera Selection
        tk.Label(self.control_frame, text="Camera Index:").pack(anchor=tk.W, padx=10)
        self.camera_var = tk.IntVar(value=0) # Default to 0 (usually the first external USB camera)
        self.camera_options = [0, 1, 2, 3, 4, 5, 6] 
        self.cam_dropdown = tk.OptionMenu(self.control_frame, self.camera_var, *self.camera_options, command=self.change_camera)
        self.cam_dropdown.config(width=15)
        self.cam_dropdown.pack(padx=10, pady=(0, 15))

        # 2. Stream Mode (Resolution / FPS)
        tk.Label(self.control_frame, text="Stream Mode:").pack(anchor=tk.W, padx=10)
        self.stream_var = tk.StringVar(value="QVGA (320x240 @ 60fps)")
        self.stream_options = [
            "VGA (640x480 @ 30fps)",
            "QVGA (320x240 @ 60fps)"
        ]
        self.stream_dropdown = tk.OptionMenu(self.control_frame, self.stream_var, *self.stream_options, command=self.change_stream_mode)
        self.stream_dropdown.config(width=20)
        self.stream_dropdown.pack(padx=10, pady=(0, 15))

        # 3. RGB Intensity Selector
        tk.Label(self.control_frame, text="LED Intensities (0-15):").pack(anchor=tk.W, padx=10)
        
        self.r_var = tk.IntVar(value=15)
        self.g_var = tk.IntVar(value=15)
        self.b_var = tk.IntVar(value=15)

        self.scale_r = tk.Scale(self.control_frame, from_=0, to=15, orient=tk.HORIZONTAL, label="Red", variable=self.r_var, command=self.update_rgb, fg="red", length=200)
        self.scale_r.pack(padx=10)
        
        self.scale_g = tk.Scale(self.control_frame, from_=0, to=15, orient=tk.HORIZONTAL, label="Green", variable=self.g_var, command=self.update_rgb, fg="green", length=200)
        self.scale_g.pack(padx=10)
        
        self.scale_b = tk.Scale(self.control_frame, from_=0, to=15, orient=tk.HORIZONTAL, label="Blue", variable=self.b_var, command=self.update_rgb, fg="blue", length=200)
        self.scale_b.pack(padx=10)

        # 4. Action Buttons
        self.btn_save = tk.Button(self.control_frame, text="Save Image", width=20, command=self.save_image, bg="lightblue", font=("Arial", 10, "bold"))
        self.btn_save.pack(pady=(30, 5), padx=10)

        self.btn_quit = tk.Button(self.control_frame, text="Quit", width=20, command=self.close_app, bg="salmon", font=("Arial", 10, "bold"))
        self.btn_quit.pack(pady=5, padx=10)

        # Status Bar
        self.status_label = tk.Label(window, text="Status: Ready to connect.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Start up the camera and the loop
        self.init_camera(self.camera_var.get())
        self.delay = 15 # Milliseconds
        self.update_frame()


    def init_camera(self, index):
        """Creates the Digit object and connects to the backend."""
        self.status_label.config(text=f"Status: Connecting to index {index}...")
        self.window.update()

        # Disconnect previous if exists
        if self.digit is not None:
            self.digit.disconnect()
            self.digit = None

        try:
            # Instantiate the new Windows Digit class
            self.digit = Digit(name="GUI_Digit", device_index=index)
            self.digit.connect()
            
            # Apply initial UI settings to the backend
            self.change_stream_mode(self.stream_var.get())
            self.update_rgb(None)
            
            self.status_label.config(text=f"Status: Connected to DIGIT at index {index}.")
        except Exception as e:
            messagebox.showwarning("Connection Error", f"Could not connect to camera at index {index}.\n\nDetails: {e}")
            self.status_label.config(text=f"Status: Error - Disconnected.")
            self.digit = None


    def change_camera(self, selection):
        """Triggered by the Camera dropdown."""
        self.init_camera(int(selection))


    def change_stream_mode(self, selection):
        """Triggered by the Stream Mode dropdown. Maps to Digit.STREAMS."""
        if self.digit:
            try:
                if "VGA" in selection and "QVGA" not in selection:
                    mode_config = self.digit.STREAMS["VGA"]
                    self.digit.set_resolution(mode_config)
                    self.digit.set_fps(mode_config["fps"]["30fps"])
                    # CHANGED: 480 wide, 640 tall
                    self.canvas.config(width=480, height=640)
                else:
                    mode_config = self.digit.STREAMS["QVGA"]
                    self.digit.set_resolution(mode_config)
                    self.digit.set_fps(mode_config["fps"]["60fps"])
                    # CHANGED: 240 wide, 320 tall
                    self.canvas.config(width=240, height=320)
                    
                self.status_label.config(text=f"Status: Stream changed to {selection}")
            except Exception as e:
                print(f"Failed to change stream mode: {e}")


    def update_rgb(self, event):
        """Triggered by sliders. Pushes RGB values to the backend."""
        if self.digit:
            r = self.r_var.get()
            g = self.g_var.get()
            b = self.b_var.get()
            try:
                self.digit.set_intensity_rgb(r, g, b)
            except Exception as e:
                print(f"Failed to update RGB: {e}")


    def update_frame(self):
        """Grabs a frame using the Digit class and updates the Tkinter canvas."""
        if self.digit:
            try:
                # The backend handles the transpose/flip and throws an Exception if it fails
                frame = self.digit.get_frame()
                
                # Save it natively for the save_image function
                self.current_frame = frame 
                
                # Convert BGR (OpenCV) to RGB (Pillow/Tkinter)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                
                # Center the image in the canvas
                canvas_w = int(self.canvas['width'])
                canvas_h = int(self.canvas['height'])
                x_offset = (canvas_w - self.photo.width()) // 2
                y_offset = (canvas_h - self.photo.height()) // 2
                
                self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor=tk.NW)
            except Exception as e:
                # Silently catch frame drop errors so the GUI doesn't crash completely
                pass

        # Loop
        self.window.after(self.delay, self.update_frame)


    def save_image(self):
        """Saves the current frame to the disk."""
        if self.current_frame is not None:
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
                
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"saved_images/digit_{timestamp}.jpg"
            
            # Using standard cv2 imwrite instead of self.digit.save_frame to guarantee
            # we save the exact frame currently rendered on the screen.
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f"Status: Saved {filename}")
            print(f"Saved: {filename}")
        else:
            messagebox.showinfo("No Stream", "No image to save. Check camera connection.")


    def close_app(self):
        """Cleanly disconnects the backend and closes the UI."""
        if self.digit:
            try:
                self.digit.disconnect()
            except:
                pass
        self.window.destroy()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x700") 
    
    app = DigitInterfaceGUI(root, "DIGIT Sensor GUI (Windows Backend)")
    
    # Clean up cleanly if the user clicks the Windows 'X' button
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()