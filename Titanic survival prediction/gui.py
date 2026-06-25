import customtkinter as ctk
import numpy as np
import pickle

# 1. Page Configuration & Theme
ctk.set_appearance_mode("Dark")  # Force a gorgeous dark dashboard
ctk.set_default_color_theme("blue")

# Load the model
with open('titanic_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Live updater for the Age Slider
def update_age_label(value):
    age_value_label.configure(text=f"{int(value)} Years")

# 2. Advanced Prediction Logic with Live UI Updates
def get_prediction():
    try:
        # Pull values from modern widgets
        pclass = int(pclass_var.get())
        age = float(age_slider.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        sex_male = 1 if sex_var.get() == "Male" else 0
        
        passenger_features = np.array([[pclass, age, sibsp, parch, sex_male]])
        prediction = loaded_model.predict(passenger_features)[0]
        
        # Calculate survival probability percentage
        probabilities = loaded_model.predict_proba(passenger_features)[0]
        
        # Dynamically transform the Result Card based on prediction
        if prediction == 1:
            confidence = probabilities[1] * 100
            result_card.configure(fg_color="#1b4d3e", border_color="#2ec4b6") # Premium Dark Green
            result_text.configure(text=f"PASSENGER SURVIVED 🎉\nConfidence: {confidence:.1f}%", text_color="#2ec4b6")
        else:
            confidence = probabilities[0] * 100
            result_card.configure(fg_color="#4a1515", border_color="#e63946") # Premium Dark Red
            result_text.configure(text=f"PASSENGER DECEASED 😢\nConfidence: {confidence:.1f}%", text_color="#e63946")
            
    except ValueError:
        result_card.configure(fg_color="#333333", border_color="#ffb703")
        result_text.configure(text="Error: Please check your inputs!", text_color="#ffb703")

# 3. Create Modern Window
root = ctk.CTk()
root.title("Titanic AI Analytics Dashboard")
root.geometry("480x620")
root.resizable(False, False)

# Main Dashboard Header
header_label = ctk.CTkLabel(root, text="TITANIC SURVIVAL PREDICTOR", font=("Segoe UI", 22, "bold"), text_color="#00b4d8")
header_label.pack(pady=(25, 5))

sub_header = ctk.CTkLabel(root, text="Machine Learning Real-Time Inference Engine", font=("Segoe UI", 11), text_color="#6c757d")
sub_header.pack(pady=(0, 15))

# Main Container Frame (The Card)
main_card = ctk.CTkFrame(root, border_width=1, border_color="#2b2b2b")
main_card.pack(pady=10, padx=25, fill="both", expand=True)

# --- Pclass (Sleek Segmented Toggle) ---
ctk.CTkLabel(main_card, text="Ticket Class (Pclass)", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=25, pady=(15, 2))
pclass_var = ctk.StringVar(value="3")
pclass_toggle = ctk.CTkSegmentedButton(main_card, values=["1", "2", "3"], variable=pclass_var, height=35)
pclass_toggle.pack(fill="x", padx=25, pady=(0, 12))

# --- Gender (Sleek Segmented Toggle) ---
ctk.CTkLabel(main_card, text="Passenger Gender", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=25, pady=(5, 2))
sex_var = ctk.StringVar(value="Male")
sex_toggle = ctk.CTkSegmentedButton(main_card, values=["Male", "Female"], variable=sex_var, height=35)
sex_toggle.pack(fill="x", padx=25, pady=(0, 12))

# --- Age (Interactive Slider with live text) ---
age_frame = ctk.CTkFrame(main_card, fg_color="transparent")
age_frame.pack(fill="x", padx=25, pady=(5, 2))
ctk.CTkLabel(age_frame, text="Passenger Age", font=("Segoe UI", 12, "bold")).pack(side="left")
age_value_label = ctk.CTkLabel(age_frame, text="28 Years", font=("Segoe UI", 12), text_color="#00b4d8")
age_value_label.pack(side="right")

age_slider = ctk.CTkSlider(main_card, from_=1, to=80, number_of_steps=79, command=update_age_label, height=18)
age_slider.set(28)
age_slider.pack(fill="x", padx=25, pady=(0, 15))

# --- SibSp & Parch (Inline Twin Layout) ---
twin_frame = ctk.CTkFrame(main_card, fg_color="transparent")
twin_frame.pack(fill="x", padx=25, pady=5)

# SibSp Column (Fixed padding here!)
sib_col = ctk.CTkFrame(twin_frame, fg_color="transparent")
sib_col.pack(side="left", expand=True, fill="x", padx=(0, 10))
ctk.CTkLabel(sib_col, text="Siblings/Spouses", font=("Segoe UI", 11, "bold")).pack(anchor="w")
sibsp_entry = ctk.CTkEntry(sib_col, placeholder_text="0", height=32, justify="center")
sibsp_entry.insert(0, "0")
sibsp_entry.pack(fill="x", pady=(2, 0))

# Parch Column
parch_col = ctk.CTkFrame(twin_frame, fg_color="transparent")
parch_col.pack(side="right", expand=True, fill="x")
ctk.CTkLabel(parch_col, text="Parents/Children", font=("Segoe UI", 11, "bold")).pack(anchor="w")
parch_entry = ctk.CTkEntry(parch_col, placeholder_text="0", height=32, justify="center")
parch_entry.insert(0, "0")
parch_entry.pack(fill="x", pady=(2, 0))

# --- Glowing Action Button ---
predict_btn = ctk.CTkButton(main_card, text="RUN INFERENCE ENGINE", font=("Segoe UI", 13, "bold"), 
                            fg_color="#0077b6", hover_color="#0096c7", height=42, command=get_prediction)
predict_btn.pack(fill="x", padx=25, pady=(25, 15))

# --- Dynamic Result Display Card ---
result_card = ctk.CTkFrame(root, height=75, fg_color="#1e1e1e", border_width=2, border_color="#2b2b2b")
result_card.pack(fill="x", padx=25, pady=(10, 25))
result_card.pack_propagate(False)

result_text = ctk.CTkLabel(result_card, text="Awaiting Input Parameters...", font=("Segoe UI", 13, "bold"), text_color="#6c757d")
result_text.pack(expand=True)

# Run UI Loop
root.mainloop()