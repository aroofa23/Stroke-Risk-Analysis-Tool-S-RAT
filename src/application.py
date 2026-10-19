import tkinter as tk
from tkinter import messagebox
import os
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from joblib import load
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the script directory
os.chdir(script_directory)

# src: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380#:~:text=The%20expected%20values%20for%20normal,and%20monitoring%20glycemia%20are%20recommended.
healthy_fasting_glucose_min = 70
healthy_fasting_glucose_max = 100

# src: https://www.cdc.gov/healthyweight/assessing/index.html#:~:text=If%20your%20BMI%20is%2018.5,falls%20within%20the%20obese%20range.
healthy_bmi_min = 18.5
healthy_bmi_max = 24.9

# Load pre-trained machine learning model and preprocessing objects
model = load("best_classifier.joblib")  # Load the trained model
scaler = load("scaler.joblib")
le_gender = load("le_gender.joblib")
le_ever_married = load("le_ever_married.joblib")
le_work_type = load("le_work_type.joblib")
le_smoking_status = load("le_smoking_status.joblib")
le_Residence_type = load("le_residence_type.joblib")

# Initialize the CSV file path
csv_file_path = "healthcare-dataset-stroke-data.csv"
resampled_dataset_path = "resampled_data.csv"

# Read the dataset
data = pd.read_csv(csv_file_path)
df = pd.read_csv(resampled_dataset_path)

# Get column names
headers = list(data.columns)


# Define the HealthApp class for managing the GUI application
class HealthApp:
    def __init__(self, master):
        # Initialize the main window
        self.master = master
        self.master.title("Stroke Risk Analysis Tool")

        # Set the geometry of the main window
        window_width = 600
        window_height = 500
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2
        self.master.geometry(
            f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}"
        )

        # Create menu bar
        self.create_menu()

        # Create widgets
        self.create_widgets()

        # Refresh patient data
        self.refresh_patient_data()

    # Method to create menu bar
    def create_menu(self):
        self.menu_bar = tk.Menu(self.master)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Create Patient File", command=self.create_patient_file
        )
        self.file_menu.add_command(label="Exit", command=self.master.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.master.config(menu=self.menu_bar)

    # Method to create widgets
    def create_widgets(self):
        # Title label
        self.label = tk.Label(
            self.master, text="Stroke Risk Analysis Tool", font=("Arial", 24, "bold")
        )
        self.label.pack(pady=20)

        self.label2 = tk.Label(
            self.master, text="Patient ID - Name", font=("Arial", 16)
        )
        self.label2.pack(pady=20)
        self.patient_data_frame = tk.Frame(self.master)
        self.patient_data_frame.pack(fill=tk.BOTH, expand=True)

    # Method to refresh patient data
    def refresh_patient_data(self):

        if os.path.exists("patient_data.csv"):
            patient_data = pd.read_csv("patient_data.csv")

            if not patient_data.empty:

                # Create the patient frame
                patient_frame = tk.Frame(self.patient_data_frame)
                patient_frame.pack()

                # Create a canvas for scrolling
                canvas = tk.Canvas(
                    patient_frame,
                    width=250,
                    highlightbackground="white",
                    highlightthickness=0,
                )
                canvas.pack(side="left", anchor="center", expand=True)

                # Create a frame inside the canvas to contain the buttons
                frame = tk.Frame(canvas, width=300)
                canvas.create_window((0, 0), window=frame)

                # Sort patient_data DataFrame by 'name' column
                sorted_patient_data = patient_data.sort_values(by="name")

                initial_button_order = []
                matching_buttons = []

                # Create a search function
                def search():
                    global matching_buttons
                    query = search_entry.get().lower()  # Get the search query
                    if query:
                        matching_buttons = [
                            button
                            for button in initial_button_order
                            if query in button.cget("text").lower()
                        ]
                    else:
                        matching_buttons = initial_button_order

                    # Repack buttons according to matching buttons
                    for button in initial_button_order:
                        if button in matching_buttons:
                            button.pack(side="top", fill="x")
                        else:
                            button.pack_forget()

                def clear_search():
                    search_entry.delete(0, tk.END)  # Clear the search entry

                    # Repack buttons in their original order
                    for button in initial_button_order:
                        button.pack(side="top", fill="x")

                # Create a frame for search entry and button
                search_frame = tk.Frame(self.patient_data_frame)
                search_frame.pack()

                # Create a search entry
                search_entry = tk.Entry(search_frame)
                search_entry.pack(side="left", padx=5, pady=5)

                # Create a search button
                search_button = tk.Button(search_frame, text="Search", command=search)
                search_button.pack(side="left", padx=5, pady=5)

                # Bind the <Return> event to the search function
                search_entry.bind("<Return>", lambda event: search())

                # Center the search frame on the screen
                search_frame.place(relx=0.5, rely=0.02, anchor="n")

                # Create a clear search button
                clear_button = tk.Button(
                    search_frame, text="Clear", command=clear_search
                )
                clear_button.pack(side="left", padx=5, pady=5)

                # Create a frame for patient buttons
                patient_frame = tk.Frame(self.patient_data_frame)
                patient_frame.pack()

                # Create a canvas for scrolling
                canvas = tk.Canvas(
                    patient_frame,
                    width=250,
                    highlightbackground="black",
                    highlightthickness=1,
                )
                canvas.pack(side="left", anchor="center", expand=True)

                # Create a frame inside the canvas to contain the buttons
                frame = tk.Frame(canvas, width=300)
                canvas.create_window((0, 0), window=frame)

                buttons = []

                # Add content to the scrollable frame in alphabetical order
                for index, row in sorted_patient_data.iterrows():
                    patient_info = f"{row['name']} - {row['id']}"
                    button = tk.Button(
                        frame,
                        text=patient_info,
                        font=("Arial", 16),
                        width=19,
                        command=lambda id=row["id"]: self.display_patient_details(id),
                    )
                    button.pack(
                        padx=5,
                        pady=5,
                        side="top",
                        anchor="center",
                        expand=True,
                        fill="x",
                    )
                    buttons.append(button)  # Add button to the list of buttons

                # Store initial button order
                initial_button_order = buttons[:]

                # Add a scrollbar
                scrollbar = tk.Scrollbar(
                    patient_frame,
                    orient=tk.VERTICAL,
                    command=canvas.yview,
                )
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                # Configure the canvas to use the scrollbar
                canvas.configure(yscrollcommand=scrollbar.set)

                # Update the canvas to reflect changes
                frame.update_idletasks()

                # Configure scroll region after scrolling to top
                canvas.config(scrollregion=canvas.bbox("all"))

                # Bind mouse wheel events for scrolling
                def on_mousewheel(event):
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

                canvas.bind_all("<MouseWheel>", on_mousewheel)

                # Center the patient frame on the screen, a bit lower
                patient_frame.place(relx=0.5, rely=0.2, anchor="n")

                # Reset the position of the scrollbar
                canvas.yview_moveto(0.0)

            else:
                messagebox.showinfo(
                    "Empty Patient Data",
                    "Please add patient details to use the app",
                )

    def show_chi_square_results(parent, feature, chi2_stat, p_value):
        # Create a new window for displaying chi-square test results
        chi_square_window = tk.Toplevel(parent)
        chi_square_window.title("Chi-Square Test Results")

        # Create labels to display the chi-square test results
        ttk.Label(chi_square_window, text=f"Feature: {feature}").pack()
        ttk.Label(chi_square_window, text=f"Chi-square statistic: {chi2_stat}").pack()
        ttk.Label(chi_square_window, text=f"p-value: {p_value}").pack()

    def show_common_info(self):
        # Create a new window for common information
        common_info_window = tk.Toplevel(self.master)
        common_info_window.title("Common Information")

        # Get the screen width and height
        screen_width = common_info_window.winfo_screenwidth()
        screen_height = common_info_window.winfo_screenheight()

        # Set the window size to a percentage of the screen size
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.9)
        common_info_window.geometry(f"{window_width}x{window_height}")

        # Calculate the maximum number of plots per row based on screen width
        max_plots_per_row = 4  # Maximum of 4 plots per row

        # Combine both numerical and categorical features
        all_features = [
            "age",
            "bmi",
            "avg_glucose_level",
            "gender",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]

        # Determine the number of rows and columns for subplots
        num_features = len(all_features)
        num_cols = min(num_features, max_plots_per_row)
        num_rows = (num_features + max_plots_per_row - 1) // max_plots_per_row

        # Calculate figure size and spacing based on window dimensions
        fig_width = window_width / num_cols / 100
        fig_height = (
            window_height / num_rows / 70
        )  # Adjusted height to fit within window
        hspace = 0.5  # Increased horizontal spacing between subplots
        wspace = 0.5  # Increased vertical spacing between subplots

        # Create a figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        # Iterate over features and create subplots
        for i, feature in enumerate(all_features):
            row_idx = i // num_cols
            col_idx = i % num_cols
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]

            if feature in ["age", "bmi", "avg_glucose_level"]:
                sns.boxplot(x="stroke", y=feature, data=df, ax=ax)
                ax.set_title(
                    f"{feature} by Stroke", fontsize=10
                )  # Decreased title font size
                ax.legend(
                    ["Non-stroke", "Stroke"],
                    loc="upper right",
                    bbox_to_anchor=(1.3, 1.5),
                )
            else:
                sns.countplot(x=feature, hue="stroke", data=df, ax=ax)
                ax.set_title(
                    f"Stroke occurrence by {feature}", fontsize=10
                )  # Decreased title font size
                ax.tick_params(
                    axis="x", rotation=45, labelsize=8
                )  # Adjusted x-axis tick label font size
                ax.tick_params(
                    axis="y", labelsize=8
                )  # Adjusted y-axis tick label font size
                ax.legend(["Non-stroke", "Stroke"], loc="upper right")

                # Perform chi-square test
                contingency_table = pd.crosstab(df[feature], df["stroke"])
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                print("Feature:", feature)
                print("Chi-square statistic:", chi2_stat)
                print("p-value:", p_value)

        # Hide any extra empty subplots
        for i in range(num_features, num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            if num_rows > 1:
                axes[row_idx, col_idx].axis("off")
            else:
                axes[col_idx].axis("off")

        # Create a canvas and pack it into the window
        canvas = FigureCanvasTkAgg(fig, master=common_info_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Update scroll region after widgets are placed in canvas
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas._tkcanvas.configure(scrollregion=canvas._tkcanvas.bbox("all"))
        plt.close("all")

    # Method to display patient details
    def display_patient_details(self, patient_id):
        # Check if patient data file exists
        if os.path.exists("patient_data.csv"):
            # Read patient data from CSV
            patient_data = pd.read_csv("patient_data.csv")
            # Extract patient info based on patient ID
            patient_info = patient_data.loc[patient_data["id"] == patient_id].squeeze()

            if not patient_info.empty:
                # Create a new window for displaying patient details
                detail_window = tk.Toplevel(self.master)
                detail_window.title(f"Patient Details - {patient_info['name']}")

                labels_and_buttons_frame = tk.Frame(
                    detail_window, highlightbackground="black", highlightthickness=1
                )
                labels_and_buttons_frame.pack(
                    side="left", fill="both", padx=10, pady=10
                )

                canvas_frame = tk.Frame(
                    detail_window, highlightbackground="black", highlightthickness=1
                )
                canvas_frame.pack(side="top", fill="both", padx=10, pady=10)

                feature_importance_frame = tk.Frame(
                    detail_window, highlightbackground="black", highlightthickness=1
                )
                feature_importance_frame.pack(side="top", fill="both", padx=10, pady=10)

                # Set window size and position
                screen_width = detail_window.winfo_screenwidth()
                screen_height = detail_window.winfo_screenheight()
                window_width = 1280
                window_height = 720
                x = (screen_width - window_width) // 2
                y = (screen_height - window_height) // 2
                detail_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

                # List of patient details to display
                details = [
                    f"Name: {patient_info['name']}",
                    f"ID: {patient_info['id']}",
                    f"Gender: {patient_info['gender_le']}",
                    f"Age: {patient_info['age']}",
                    f"Hypertension: {patient_info['hypertension']}",
                    f"Heart Disease: {patient_info['heart_disease']}",
                    f"Ever Married: {patient_info['ever_married_le']}",
                    f"Work Type: {patient_info['work_type_le']}",
                    f"Residence Type: {patient_info['residence_type_le']}",
                    f"Average Glucose Level: {patient_info['avg_glucose_level']}",
                    f"BMI: {patient_info['bmi']}",
                    f"Smoking Status: {patient_info['smoking_status_le']}",
                ]

                # Display patient details
                for detail in details:
                    tk.Label(
                        labels_and_buttons_frame, text=detail, font=("Arial", 12)
                    ).pack(
                        anchor="w", padx=10, pady=5
                    )  # aligned text to left for space for graph displays

                # Prepare features for prediction
                features = np.array(
                    [
                        [
                            patient_info["age"],
                            patient_info["hypertension"],
                            patient_info["heart_disease"],
                            patient_info["avg_glucose_level"],
                            patient_info["bmi"],
                            le_gender.transform([patient_info["gender_le"]])[0],
                            le_ever_married.transform(
                                [patient_info["ever_married_le"]]
                            )[0],
                            le_work_type.transform([patient_info["work_type_le"]])[0],
                            le_smoking_status.transform(
                                [patient_info["smoking_status_le"]]
                            )[0],
                            le_Residence_type.transform(
                                [patient_info["residence_type_le"]]
                            )[0],
                        ]
                    ],
                )
                
                # Scale the features using the pre-trained scaler
                features_scaled = scaler.transform(features)
                print(features_scaled)

                # Define feature names for creating a DataFrame
                feature_names = [
                    "age",
                    "hypertension",
                    "heart_disease",
                    "avg_glucose_level",
                    "bmi",
                    "gender_le",
                    "ever_married_le",
                    "work_type_le",
                    "smoking_status_le",
                    "residence_type_le",
                ]
                features_scaled_df = pd.DataFrame(
                    features_scaled, columns=feature_names
                )

                features_scaled_df.to_csv("feature.csv", index=False)
                # Predict stroke risk
                stroke_prediction = model.predict(features_scaled_df)
                prediction_probability = (
                    model.predict_proba(features_scaled_df)[:, 1] * 100
                )

                # Display stroke prediction result
                prediction_text = f"Stroke Risk: {'Yes' if stroke_prediction[0] == 1 else 'No'} ({prediction_probability[0]:.2f}% probability of being YES)"
                tk.Label(
                    labels_and_buttons_frame,
                    text=prediction_text,
                    font=("Arial", 14, "bold"),
                ).pack(
                    anchor="w", padx=10, pady=5
                )  # aligned buttons to left for space for graph displays
                
                # Create buttons for editing information, treatment recommendations, and showing common info
                edit_info_button = tk.Button(
                    labels_and_buttons_frame,
                    text="Edit Information",
                    font=("Arial", 12),
                    width=25,
                    bg="green",
                    command=lambda: (
                        detail_window.destroy(),
                        self.on_edit_click(patient_id),
                    ),
                )

                tr_button = tk.Button(
                    labels_and_buttons_frame,
                    text="Treatment Recommendations",
                    font=("Arial", 12),
                    width=25,
                    bg="green",
                    command=lambda: self.on_TR_click(patient_id, stroke_prediction),
                )

                show_common_button = tk.Button(
                    labels_and_buttons_frame,
                    text="Show Common Info",
                    font=("Arial", 12),
                    width=25,
                    bg="green",
                    command=self.show_common_info,
                )

                # Pack buttons into the frame
                edit_info_button.pack(
                    side="top", anchor="center", padx=10, pady=(15, 25)
                )
                tr_button.pack(side="top", anchor="center", padx=10, pady=(15, 25))
                show_common_button.pack(
                    side="top", anchor="center", padx=10, pady=(15, 25)
                )

                # obtain patient info for graph display
                glucose_level = patient_info["avg_glucose_level"]
                bmi = patient_info["bmi"]

                print(glucose_level)
                print(bmi)

                # Create a frame to contain the canvas
                canvas_frame = tk.Frame(canvas_frame)
                canvas_frame.pack(fill="both", expand=True)

                # Create the canvas
                canvas = tk.Canvas(
                    canvas_frame,
                    width=600,
                    height=150,
                    highlightthickness=1,
                    highlightbackground="black",
                )
                canvas.pack(fill="both", expand=True)  # Pack canvas into canvas_frame

                # Draw the number line for glucose levels
                canvas.create_line(20, 40, 580, 40, width=5)  # Line for glucose levels
                canvas.create_text(
                    20, 30, text="50", anchor=tk.W, font=("Arial", 8)
                )  # Text for glucose min
                canvas.create_text(
                    580, 30, text="250", anchor=tk.E, font=("Arial", 8)
                )  # Text for glucose max
                canvas.create_text(
                    300,
                    10,
                    text="Glucose Levels",
                    anchor=tk.N,
                    font=("Arial", 12, "bold"),
                )  # Title

                # Draw highlighted range for healthy glucose levels
                canvas.create_rectangle(
                    20 + (70 - 50) / (250 - 50) * 560,
                    30,
                    20 + (100 - 50) / (250 - 50) * 560,
                    40,
                    fill="green",
                )
                # Add labels for healthy range start and end values
                canvas.create_text(
                    20 + (70 - 50) / (250 - 50) * 560,
                    50,
                    text="70",
                    anchor=tk.N,
                    font=("Arial", 8),
                )
                canvas.create_text(
                    20 + (100 - 50) / (250 - 50) * 560,
                    50,
                    text="100",
                    anchor=tk.N,
                    font=("Arial", 8),
                )

                # Draw a marker for patient's glucose level
                glucose_marker_position = 20 + (glucose_level - 50) / (250 - 50) * 560
                canvas.create_line(
                    glucose_marker_position,
                    30,
                    glucose_marker_position,
                    40,
                    width=6,
                    fill="red",
                )

                # Draw the number line for BMI
                canvas.create_line(20, 90, 580, 90, width=5)  # Line for BMI values
                canvas.create_text(
                    20, 80, text="0", anchor=tk.W, font=("Arial", 8)
                )  # Text for BMI min
                canvas.create_text(
                    580, 80, text="50", anchor=tk.E, font=("Arial", 8)
                )  # Text for BMI max
                canvas.create_text(
                    300, 70, text="BMI Value", anchor=tk.N, font=("Arial", 12, "bold")
                )  # Title

                # Draw highlighted range for healthy BMI values
                canvas.create_rectangle(
                    20 + (18.5 - 0) / (50 - 0) * 560,
                    90,
                    20 + (24.9 - 0) / (50 - 0) * 560,
                    100,
                    fill="green",
                )
                # Add labels for healthy range start and end values
                canvas.create_text(
                    20 + (18.5 - 0) / (50 - 0) * 560,
                    110,
                    text="18.5",
                    anchor=tk.N,
                    font=("Arial", 8),
                )
                canvas.create_text(
                    20 + (24.9 - 0) / (50 - 0) * 560,
                    110,
                    text="24.9",
                    anchor=tk.N,
                    font=("Arial", 8),
                )

                # Draw a marker for patient's BMI
                bmi_marker_position = 20 + (bmi - 0) / (50 - 0) * 560
                canvas.create_line(
                    bmi_marker_position,
                    90,
                    bmi_marker_position,
                    100,
                    width=6,
                    fill="blue",
                )

                # Draw legend
                canvas.create_rectangle(20, 125, 30, 135, fill="green")
                canvas.create_text(
                    40, 130, text="Healthy Range", anchor=tk.W, font=("Arial", 8)
                )

                # Check if the model is a tree-based model
                if hasattr(model, "feature_importances_"):
                    # Get feature importances
                    feature_importances = model.feature_importances_

                    # Create a figure and axis for the plot
                    fig, ax = plt.subplots(figsize=(6, 6))

                    # Create a bar plot
                    ax.bar(
                        range(len(feature_importances)),
                        feature_importances,
                        color="green",
                    )
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45)
                    ax.set_xlabel("Features")
                    ax.set_ylabel("Significance")
                    ax.set_title("Feature significance in predicting a stroke")

                    # Adjust subplot spacing to make plot smaller within the frame
                    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

                    # Create a canvas to display the plot in the GUI
                    canvas = FigureCanvasTkAgg(fig, master=feature_importance_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    plt.close("all")

                else:
                    messagebox.showinfo(
                        "Error", "The feature importance plot can not be genertaed."
                    )

            else:
                messagebox.showinfo("Error", "No data found for this patient.")
        else:
            messagebox.showinfo("Error", "Patient data file not found.")

    def on_TR_click(self, patient_id, stroke_prediction):

        # Initialize an empty list to store treatment recommendations
        recommendations = []

        # Create a new window for displaying treatment recommendations
        TR_window = tk.Toplevel(self.master)
        TR_window.title("Treatment Recommendations")
        TR_window.geometry("800x600")
        screen_width = TR_window.winfo_screenwidth()
        screen_height = TR_window.winfo_screenheight()
        window_width = 800
        window_height = 600

        # Calculate the position to center the window on the screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set the geometry of the window to center it on the screen
        TR_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Read patient data from CSV file
        patient_data = pd.read_csv("patient_data.csv")
        patient_info = patient_data.loc[patient_data["id"] == patient_id].squeeze()

        # Check for hypertension and add relevant recommendation
        if patient_info["hypertension"] == 1:
            recommendations.append("Hypertension:")
            recommendations.append(
                "Having a history of hypertension can subject you to having a higher risk of a stroke. Managing hypertension involves adopting a healthy lifestyle, including a balanced diet rich in fruits, vegetables, and whole grains, regular exercise, weight management, limited alcohol consumption, and quitting smoking. Stress reduction techniques such as deep breathing exercises and meditation are also beneficial. Monitoring blood pressure regularly at home and attending regular check-ups with a healthcare provider are essential. In some cases, medication may be prescribed to lower blood pressure. It's crucial to follow the treatment plan recommended by your healthcare provider to effectively manage hypertension and reduce the risk of complications such as stroke and heart disease."
            )

        # Check for heart disease and add relevant recommendation
        if patient_info["heart_disease"] == 1:
            recommendations.append("Heart Disease:")
            recommendations.append(
                "Having a history of heart disease can subject you to having a higher risk of a stroke. Managing heart disease requires a multifaceted approach. Lifestyle changes such as adopting a heart-healthy diet low in saturated fats, cholesterol, and sodium, engaging in regular physical activity, maintaining a healthy weight, quitting smoking, and managing stress are fundamental. Medications may be prescribed to control risk factors such as high blood pressure, high cholesterol, and diabetes. Regular medical check-ups, including monitoring blood pressure, cholesterol levels, and other relevant tests, are essential for tracking progress and adjusting treatment plans as needed. In some cases, procedures or surgeries may be recommended to improve blood flow to the heart or treat underlying conditions. Compliance with prescribed medications, lifestyle modifications, and ongoing communication with healthcare providers are critical for effectively managing heart disease and reducing the risk of complications."
            )

        # Check glucose level and add relevant recommendation
        if patient_info["avg_glucose_level"] > healthy_fasting_glucose_max:
            recommendations.append("High Glucose:")
            recommendations.append(
                "Having a high glucose level can subject you to having a higher risk of a stroke. Yes, there is a correlation between glucose levels and stroke risk. High glucose levels, especially over a prolonged period, can contribute to the development of conditions such as diabetes and metabolic syndrome. These conditions are associated with an increased risk of stroke. High glucose levels can lead to damage to blood vessels and promote inflammation, which are both factors that contribute to the development of atherosclerosis (hardening and narrowing of the arteries), a leading cause of stroke. Additionally, high glucose levels can also increase the risk of blood clot formation, further raising the likelihood of stroke occurrence. Therefore, managing and controlling glucose levels through lifestyle modifications and, if necessary, medication can be essential in reducing the risk of stroke."
            )

        # Check glucose level and add relevant recommendation
        if patient_info["avg_glucose_level"] < healthy_fasting_glucose_min:
            recommendations.append("Low Glucose:")
            recommendations.append(
                "Having a low glucose level can subject you to having a higher risk of a stroke. Yes, there is a correlation between glucose levels and stroke risk. Having low glucose levels, or hypoglycemia, can lead to symptoms such as dizziness, weakness, confusion, and in severe cases, loss of consciousness. If you experience these symptoms, it's crucial to take immediate action to raise your blood sugar levels. Consume fast-acting carbohydrates such as fruit juice, glucose tablets, or candy to quickly increase your glucose levels. Follow this with a snack or meal containing protein and complex carbohydrates to stabilize your blood sugar. Monitor your blood sugar levels regularly and discuss any concerns with your healthcare provider. Adjustments to your diet, medication, or activity level may be necessary to prevent future episodes of hypoglycemia and maintain stable blood sugar levels to reduce the risk of stroke."
            )

        if patient_info["bmi"] > healthy_bmi_max:
            recommendations.append("High BMI:")
            recommendations.append(
                "Having a high BMI can subject you to having a higher risk of a stroke. To lower BMI, focus on a balanced diet with plenty of fruits, vegetables, lean proteins, and whole grains while limiting processed foods and sugary snacks. Incorporate regular exercise, aiming for at least 150 minutes of moderate-intensity activity per week. Make small lifestyle changes like taking the stairs or standing more throughout the day. Stay hydrated, prioritize quality sleep, and seek support from friends, family, or healthcare professionals. Monitoring progress and setting realistic goals are key to achieving sustainable weight loss and a healthier BMI."
            )

        # Check BMI and add relevant recommendation
        if patient_info["bmi"] < healthy_bmi_min:
            recommendations.append("Low BMI:")
            recommendations.append(
                "Having a low BMI can subject you to having a higher risk of a stroke. To raise BMI, focus on increasing calorie intake through nutrient-dense foods such as healthy fats, lean proteins, complex carbohydrates, and plenty of fruits and vegetables. Incorporate snacks between meals and larger portion sizes during meals. Include resistance training exercises to build muscle mass, which can contribute to weight gain. Reduce the intensity or duration of cardiovascular exercise to minimize calorie expenditure. Consult with a healthcare provider or registered dietitian for personalized guidance on safe and effective ways to increase BMI."
            )

        # Check smoking status and add relevant recommendation
        if patient_info["smoking_status_le"]:
            recommendations.append("Smoking:")
            recommendations.append(
                "Smoking can subject you to having a higher risk of a stroke. To quit smoking, start by setting a quit date and making a plan. Identify triggers and develop coping strategies, such as keeping hands busy or practicing deep breathing. Consider using nicotine replacement therapy (NRT) or prescription medications to manage withdrawal symptoms. Seek support from friends, family, or a support group, and consider counseling or behavioral therapy. Remove smoking paraphernalia and create a smoke-free environment. Stay committed, stay positive, and remember that quitting is a process – if setbacks occur, keep trying."
            )

        # Ensure that recommendations are present
        assert recommendations

        # Create a text widget to display recommendations
        text_widget = tk.Text(
            TR_window, wrap=tk.WORD, width=60, height=30, font=("Arial", 12)
        )
        text_widget.pack(expand=True, fill=tk.BOTH)

        # Iterate over recommendations and insert them into the text widget
        for recommendation in recommendations:
            # Check if the line starts with a header (ending with ':')
            if recommendation.endswith(":"):
                text_widget.insert(tk.END, recommendation + "\n", "bold")
            else:
                text_widget.insert(tk.END, recommendation + "\n\n")

        # Disable editing in the text widget
        text_widget.config(state=tk.DISABLED)

        # Configure the 'bold' tag to display bold text
        text_widget.tag_configure("bold", font=("Arial", 12, "bold"))

        print_button = tk.Button(TR_window, text="Print")
        print_button.pack(side=tk.RIGHT, anchor=tk.SE, padx=10, pady=10)

    # Other methods like create_patient_file remain mostly unchanged...
    def on_edit_click(self, patient_id):

        # Function to handle editing of patient information
        def save_changes():
            # Retrieve updated information from the entry fields
            updated_patient_info = {
                "name": name_entry.get(),
                "id": id_entry.get(),
                "gender_le": gender_var.get(),
                "age": age_entry.get(),
                "hypertension": hypertension_var.get(),
                "heart_disease": heart_disease_var.get(),
                "ever_married_le": ever_married_var.get(),
                "work_type_le": work_type_var.get(),
                "residence_type_le": residence_var.get(),
                "avg_glucose_level": avg_glucose_level_entry.get(),
                "bmi": bmi_entry.get(),
                "smoking_status_le": smoking_var.get(),
            }

            # Check if the name field is empty
            if not name_entry.get().strip():
                messagebox.showerror("Error", "Please enter a Name.")
                return

            # Check if the ID field is empty
            if not id_entry.get().strip():
                messagebox.showerror("Error", "Please enter a Patient ID.")
                return

            # Check if the age field is empty
            if not age_entry.get().strip():
                messagebox.showerror("Error", "Please enter an Age.")
                return

            # Check if the glucose level field is empty
            if not avg_glucose_level_entry.get().strip():
                messagebox.showerror("Error", "Please enter a Glucose Level.")
                return

            # Check if the BMI field is empty
            if not bmi_entry.get().strip():
                messagebox.showerror("Error", "Please enter a BMI.")
                return

            id = int(id_entry.get())

            # Check if age is numerical
            age = int(age_entry.get())

            # Check if the average glucose level is numerical
            avg_glucose_level = float(avg_glucose_level_entry.get())

            # Check if BMI is numerical
            bmi = float(bmi_entry.get())

            # Update the corresponding patient record in the CSV file
            patient_data = pd.read_csv("patient_data.csv")
            patient_index = patient_data.index[patient_data["id"] == patient_id][0]

            for key, value in updated_patient_info.items():
                patient_data.at[patient_index, key] = value

            patient_data.to_csv("patient_data.csv", index=False)

            messagebox.showinfo("Success", "Patient information updated successfully")
            edit_window.destroy()
            # Optionally, update the displayed patient details in the main window
            self.display_patient_details(patient_id)

        def on_edit_window_close():
            self.display_patient_details(patient_id)
            edit_window.destroy()

        # Create a new window for editing patient information
        edit_window = tk.Toplevel(self.master)
        edit_window.title("Edit Patient Information")

        screen_width = edit_window.winfo_screenwidth()
        screen_height = edit_window.winfo_screenheight()

        window_width = 350
        window_height = 350

        # Calculate the x and y coordinates to center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set the geometry of the window to center it on the screen
        edit_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        edit_window.protocol("WM_DELETE_WINDOW", on_edit_window_close)

        # Fetch current patient information
        patient_data = pd.read_csv("patient_data.csv")
        patient_info = patient_data.loc[patient_data["id"] == patient_id].squeeze()

        # Populate the form fields with current patient information
        tk.Label(edit_window, text="Name:").grid(row=0, column=0)
        name_entry = tk.Entry(edit_window)
        name_entry.grid(row=0, column=1)
        name_entry.insert(0, patient_info["name"])

        tk.Label(edit_window, text="ID:").grid(row=1, column=0)
        id_entry = tk.Entry(edit_window)
        id_entry.grid(row=1, column=1)
        id_entry.insert(0, patient_info["id"])

        # Define options for dropdown menus
        gender_options = ["Male", "Female"]
        yes_no_options = ["Yes", "No"]
        yes_no_1_0_option = [1, 0]
        residence_options = ["Rural", "Urban"]
        work_type_options = [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ]
        smoking_status_options = [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown",
        ]

        # Populate option menus with current patient information
        tk.Label(edit_window, text="Gender:").grid(row=2, column=0)
        gender_var = tk.StringVar(edit_window)
        gender_var.set(patient_info["gender_le"])  # Set default value
        gender_dropdown = tk.OptionMenu(edit_window, gender_var, *gender_options)
        gender_dropdown.grid(row=2, column=1)

        # Label and entry for Age
        tk.Label(edit_window, text="Age:").grid(row=3, column=0)
        age_entry = tk.Entry(edit_window)
        age_entry.grid(row=3, column=1)
        age_entry.insert(0, patient_info["age"])

        # Dropdown menu for Hypertension
        tk.Label(edit_window, text="Hypertension:").grid(row=4, column=0)
        hypertension_var = tk.StringVar(edit_window)
        hypertension_var.set(patient_info["hypertension"])  # Set default value
        hypertension_dropdown = tk.OptionMenu(
            edit_window, hypertension_var, *yes_no_1_0_option
        )
        hypertension_dropdown.grid(row=4, column=1)

        # Dropdown menu for Heart Disease
        tk.Label(edit_window, text="Heart Disease:").grid(row=5, column=0)
        heart_disease_var = tk.StringVar(edit_window)
        heart_disease_var.set(patient_info["heart_disease"])  # Set default value
        heart_disease_dropdown = tk.OptionMenu(
            edit_window, heart_disease_var, *yes_no_1_0_option
        )
        heart_disease_dropdown.grid(row=5, column=1)

        # Dropdown menu for Ever Married
        tk.Label(edit_window, text="Ever Married:").grid(row=6, column=0)
        ever_married_var = tk.StringVar(edit_window)
        ever_married_var.set(patient_info["ever_married_le"])  # Set default value
        ever_married_dropdown = tk.OptionMenu(
            edit_window, ever_married_var, *yes_no_options
        )
        ever_married_dropdown.grid(row=6, column=1)

        # Dropdown menu for Work Type
        tk.Label(edit_window, text="Work Type:").grid(row=7, column=0)
        work_type_var = tk.StringVar(edit_window)
        work_type_var.set(patient_info["work_type_le"])  # Set default value
        work_type_dropdown = tk.OptionMenu(
            edit_window, work_type_var, *work_type_options
        )
        work_type_dropdown.grid(row=7, column=1)

        # Dropdown menu for Residence Type
        tk.Label(edit_window, text="Residence Type:").grid(row=8, column=0)
        residence_var = tk.StringVar(edit_window)
        residence_var.set(patient_info["residence_type_le"])  # Set default value
        residence_dropdown = tk.OptionMenu(
            edit_window, residence_var, *residence_options
        )
        residence_dropdown.grid(row=8, column=1)

        # Entry for Average Glucose Level
        tk.Label(edit_window, text="Average Glucose Level:").grid(row=9, column=0)
        avg_glucose_level_entry = tk.Entry(edit_window)
        avg_glucose_level_entry.grid(row=9, column=1)
        avg_glucose_level_entry.insert(0, patient_info["avg_glucose_level"])

        # Entry for BMI
        tk.Label(edit_window, text="BMI:").grid(row=10, column=0)
        bmi_entry = tk.Entry(edit_window)
        bmi_entry.grid(row=10, column=1)
        bmi_entry.insert(0, patient_info["bmi"])

        # Dropdown menu for Smoking Status
        tk.Label(edit_window, text="Smoking Status:").grid(row=11, column=0)
        smoking_var = tk.StringVar(edit_window)
        smoking_var.set(patient_info["smoking_status_le"])  # Set default value
        smoking_dropdown = tk.OptionMenu(
            edit_window, smoking_var, *smoking_status_options
        )
        smoking_dropdown.grid(row=11, column=1)

        # Add a "Save" button to save changes
        save_button = tk.Button(edit_window, text="Save", command=save_changes)
        save_button.grid(row=12, columnspan=2)

    def create_patient_file(self):
        # Create a new window
        window = tk.Toplevel(self.master)
        window.title("Enter Patient Data")

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Set the width and height of the window
        window_width = 350
        window_height = 350

        # Calculate the x and y coordinates to center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set the geometry of the window to center it on the screen
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Define options for dropdowns
        gender_options = ["Male", "Female"]
        yes_no_options = ["Yes", "No"]
        yes_no_1_0_option = [1, 0]
        residence_options = ["Rural", "Urban"]
        work_type_options = [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ]
        smoking_status_options = [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown",
        ]

        # Label and entry for Patient Name
        tk.Label(window, text="Patient Name:").grid(row=0, column=0)
        name_entry = tk.Entry(window)
        name_entry.grid(row=0, column=1)

        # Label and entry for Patient ID
        tk.Label(window, text="Patient ID:").grid(row=1, column=0)
        id_entry = tk.Entry(window)
        id_entry.grid(row=1, column=1)

        # Create labels and dropdowns for each field
        tk.Label(window, text="Gender:").grid(row=2, column=0)
        gender_var = tk.StringVar(window)
        gender_var.set(gender_options[0])  # default value
        gender_dropdown = tk.OptionMenu(window, gender_var, *gender_options)
        gender_dropdown.grid(row=2, column=1)

        # Label and entry for Age
        tk.Label(window, text="Age:").grid(row=3, column=0)
        age_entry = tk.Entry(window)
        age_entry.grid(row=3, column=1)

        # Label and dropdown for Hypertension
        tk.Label(window, text="Hypertension:").grid(row=4, column=0)
        hypertension_var = tk.StringVar(window)
        hypertension_var.set(yes_no_1_0_option[0])  # default value
        hypertension_dropdown = tk.OptionMenu(
            window, hypertension_var, *yes_no_1_0_option
        )
        hypertension_dropdown.grid(row=4, column=1)

        # Label and dropdown for Heart Disease
        tk.Label(window, text="Heart Disease:").grid(row=5, column=0)
        heart_disease_var = tk.StringVar(window)
        heart_disease_var.set(yes_no_1_0_option[0])  # default value
        heart_disease_dropdown = tk.OptionMenu(
            window, heart_disease_var, *yes_no_1_0_option
        )
        heart_disease_dropdown.grid(row=5, column=1)

        # Label and dropdown for Ever Married
        tk.Label(window, text="Ever Married:").grid(row=6, column=0)
        ever_married_var = tk.StringVar(window)
        ever_married_var.set(yes_no_options[0])  # default value
        ever_married_dropdown = tk.OptionMenu(window, ever_married_var, *yes_no_options)
        ever_married_dropdown.grid(row=6, column=1)

        # Label and dropdown for Work Type
        tk.Label(window, text="Work Type:").grid(row=7, column=0)
        work_type_entry = tk.StringVar(window)
        work_type_entry.set(work_type_options[0])
        work_type_dropdown = tk.OptionMenu(window, work_type_entry, *work_type_options)
        work_type_dropdown.grid(row=7, column=1)

        # Label and dropdown for Residence Type
        tk.Label(window, text="Residence Type:").grid(row=8, column=0)
        residence_var = tk.StringVar(window)
        residence_var.set(residence_options[0])  # default value
        residence_dropdown = tk.OptionMenu(window, residence_var, *residence_options)
        residence_dropdown.grid(row=8, column=1)

        # Label and entry for Average Glucose Level
        tk.Label(window, text="Average Glucose Level:").grid(row=9, column=0)
        avg_glucose_level_entry = tk.Entry(window)
        avg_glucose_level_entry.grid(row=9, column=1)

        # Label and entry for BMI
        tk.Label(window, text="BMI:").grid(row=10, column=0)
        bmi_entry = tk.Entry(window)
        bmi_entry.grid(row=10, column=1)

        tk.Label(window, text="Smoking Status:").grid(row=11, column=0)
        smoking_var = tk.StringVar(window)
        smoking_var.set(smoking_status_options[0])  # default value
        smoking_dropdown = tk.OptionMenu(window, smoking_var, *smoking_status_options)
        smoking_dropdown.grid(row=11, column=1)

        # Function to save patient data
        def save_data():
            try:

                # Check if name field is empty
                if not name_entry.get().strip():
                    messagebox.showerror("Error", "Please enter a Name.")
                    return

                # Check if ID field is empty
                if not id_entry.get().strip():
                    messagebox.showerror("Error", "Please enter a Patient ID.")
                    return

                # Check if age field is empty
                if not age_entry.get().strip():
                    messagebox.showerror("Error", "Please enter an Age.")
                    return

                # Check if glucose level field is empty
                if not avg_glucose_level_entry.get().strip():
                    messagebox.showerror("Error", "Please enter a Glucose Level.")
                    return

                # Check if BMI field is empty
                if not bmi_entry.get().strip():
                    messagebox.showerror("Error", "Please enter a BMI.")
                    return

                id = int(id_entry.get())

                # Check if age is numerical
                age = int(age_entry.get())

                # Check if average glucose level is numerical
                avg_glucose_level = float(avg_glucose_level_entry.get())

                # Check if BMI is numerical
                bmi = float(bmi_entry.get())

                # Extracting data from the entry fields
                age = int(age_entry.get())
                avg_glucose_level = float(avg_glucose_level_entry.get())
                bmi = float(bmi_entry.get())

                # Check if ID already exists
                if check_duplicate_id(id_entry.get()):
                    messagebox.showerror("Error", "Patient ID already exists.")
                    return

                # Creating a dictionary to store patient data
                patient_data = {
                    "id": id_entry.get(),
                    "name": name_entry.get(),
                    "age": age,
                    "hypertension": hypertension_var.get(),
                    "heart_disease": heart_disease_var.get(),
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "gender_le": gender_var.get(),
                    "ever_married_le": ever_married_var.get(),
                    "work_type_le": work_type_entry.get(),
                    "smoking_status_le": smoking_var.get(),
                    "residence_type_le": residence_var.get(),
                }
                # Checking if the file exists, then appending data
                file_exists = os.path.isfile("patient_data.csv")
                with open("patient_data.csv", "a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=patient_data.keys())
                    if not file_exists:  # Check if file is empty
                        writer.writeheader()
                    writer.writerow(patient_data)

                # Refreshing patient data and displaying success message
                self.refresh_patient_data()
                messagebox.showinfo("Patient File", "Patient data added successfully.")
                window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid data.")

        def check_duplicate_id(new_id):
            if os.path.isfile("patient_data.csv"):
                with open("patient_data.csv", "r") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["id"] == new_id:
                            return True
            return False

        # Create a button to save data
        save_button = tk.Button(window, text="Save", command=save_data)
        save_button.grid(row=12, columnspan=2)


def main():
    root = tk.Tk()
    app = HealthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
