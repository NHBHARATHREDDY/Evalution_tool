import gradio as gr
import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd

# Load dataset
file_path = "Student-Employability-Datasets.xlsx"
df = pd.read_excel(file_path, sheet_name="Data")

# Prepare training data
features = [
    "GENERAL APPEARANCE", "MANNER OF SPEAKING", "PHYSICAL CONDITION",
    "MENTAL ALERTNESS", "SELF-CONFIDENCE", "ABILITY TO PRESENT IDEAS", "COMMUNICATION SKILLS"
]
X = df[features].values
y = (df["CLASS"] == "Employable").astype(int)  # Convert to binary labels

# Train Perceptron Model
model = Perceptron()
model.fit(X, y)

def assess_employability(name, *ratings):
    user_input = np.array(ratings).reshape(1, -1)
    prediction = model.predict(user_input)[0]
    
    if prediction == 1:
        return f"Congrats {name}!!! 🎉 You are employable."
    else:
        return f"{name}, try to upgrade yourself! 📚 Keep learning and improving."

# Create UI
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Employability Assessment")
        name = gr.Textbox(label="Enter Your Name")
        sliders = [gr.Slider(1, 5, value=3, label=feature) for feature in features]
        submit = gr.Button("Get Yourself Evaluated")
        output = gr.Textbox(label="Result")
        
        submit.click(assess_employability, inputs=[name] + sliders, outputs=output)
    
    return demo

app = create_ui()
app.launch()
