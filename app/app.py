import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# 1. FIND PROJECT ROOT AUTOMATICALLY
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# app.py is inside:
# student-performance-predictor/app/app.py
#
# Therefore project root is one level above app/

PROJECT_DIR = os.path.dirname(CURRENT_DIR)


# =========================================================
# 2. FIND MODELS DIRECTORY
# =========================================================

possible_model_dirs = [
    os.path.join(PROJECT_DIR, "models"),
    os.path.join(CURRENT_DIR, "models"),
]

MODELS_DIR = None

for folder in possible_model_dirs:
    if os.path.isdir(folder):
        MODELS_DIR = folder
        break


# If models folder cannot be found
if MODELS_DIR is None:

    st.error("❌ MODELS FOLDER NOT FOUND")

    st.write("Current app location:")
    st.code(CURRENT_DIR)

    st.write("Project location:")
    st.code(PROJECT_DIR)

    st.write("Python is searching these locations:")

    for folder in possible_model_dirs:
        st.code(folder)

    st.stop()


# =========================================================
# 3. FIND DATA DIRECTORY
# =========================================================

possible_data_dirs = [
    os.path.join(PROJECT_DIR, "data"),
    os.path.join(CURRENT_DIR, "data"),
]

DATA_DIR = None

for folder in possible_data_dirs:
    if os.path.isdir(folder):
        DATA_DIR = folder
        break


# Create data directory if needed

if DATA_DIR is None:

    DATA_DIR = os.path.join(
        PROJECT_DIR,
        "data"
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )


# =========================================================
# 4. FUNCTION TO FIND FILE
# =========================================================

def find_file(folder, possible_names):

    for name in possible_names:

        path = os.path.join(
            folder,
            name
        )

        if os.path.isfile(path):

            return path

    return None


# =========================================================
# 5. FIND MODEL FILE
# =========================================================

model_candidates = [

    "student_performance_final_model.pkl",

    "student_performance_model.pkl",

    "student_performance.pkl",

    "final_model.pkl",

    "model.pkl",

    "best_model.pkl",

    "student_performance_final_model.joblib",

    "student_performance_model.joblib",

    "model.joblib",

]


MODEL_PATH = find_file(
    MODELS_DIR,
    model_candidates
)


# =========================================================
# 6. AUTOMATIC MODEL SEARCH
# =========================================================

if MODEL_PATH is None:

    all_model_files = []

    for filename in os.listdir(MODELS_DIR):

        lower_name = filename.lower()

        if (
            lower_name.endswith(".pkl")
            or lower_name.endswith(".joblib")
            or lower_name.endswith(".pkl.gz")
            or lower_name.endswith(".joblib.gz")
        ):

            # Do not use metrics/preprocessor as prediction model

            if not any(
                word in lower_name
                for word in [
                    "metric",
                    "confusion",
                    "preprocessor",
                    "comparison",
                    "history"
                ]
            ):

                all_model_files.append(
                    filename
                )


    if len(all_model_files) > 0:

        MODEL_PATH = os.path.join(
            MODELS_DIR,
            all_model_files[0]
        )


# =========================================================
# 7. FIND OTHER FILES
# =========================================================

METRICS_PATH = find_file(
    MODELS_DIR,
    [
        "model_metrics.pkl",
        "metrics.pkl",
        "model_metrics.joblib"
    ]
)


CONFUSION_MATRIX_PATH = find_file(
    MODELS_DIR,
    [
        "confusion_matrix.pkl",
        "confusion_matrix.joblib"
    ]
)


MODEL_COMPARISON_PATH = find_file(
    MODELS_DIR,
    [
        "model_comparison_results.csv",
        "model_comparison.csv",
        "model_comparison_results.pkl"
    ]
)


PREPROCESSOR_PATH = find_file(
    MODELS_DIR,
    [
        "preprocessor.pkl",
        "preprocessor.joblib"
    ]
)


HISTORY_PATH = os.path.join(
    DATA_DIR,
    "prediction_history.csv"
)


# =========================================================
# 8. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# 9. MODEL LOADING
# =========================================================

if MODEL_PATH is None:

    st.error("❌ MODEL LOADING ERROR")

    st.write(
        "No trained model file was found."
    )

    st.write("Models folder being searched:")

    st.code(
        MODELS_DIR
    )

    st.write("Files currently inside models folder:")

    try:

        files = os.listdir(
            MODELS_DIR
        )

        for file in files:

            st.write(
                "📄 " + file
            )

    except Exception as e:

        st.error(
            str(e)
        )

    st.stop()


try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as e:

    st.error(
        "❌ MODEL FOUND BUT COULD NOT BE LOADED"
    )

    st.write(
        "Model path:"
    )

    st.code(
        MODEL_PATH
    )

    st.write(
        "Error:"
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# 10. LOAD PREPROCESSOR
# =========================================================

preprocessor = None

if PREPROCESSOR_PATH is not None:

    try:

        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

    except Exception:

        preprocessor = None


# =========================================================
# 11. LOAD METRICS
# =========================================================

metrics = {}

if METRICS_PATH is not None:

    try:

        metrics = joblib.load(
            METRICS_PATH
        )

    except Exception:

        metrics = {}


# =========================================================
# 12. LOAD CONFUSION MATRIX
# =========================================================

cm = None

if CONFUSION_MATRIX_PATH is not None:

    try:

        cm = joblib.load(
            CONFUSION_MATRIX_PATH
        )

    except Exception:

        cm = None


# =========================================================
# 13. LOAD MODEL COMPARISON
# =========================================================

model_comparison = None

if MODEL_COMPARISON_PATH is not None:

    try:

        if MODEL_COMPARISON_PATH.endswith(".csv"):

            model_comparison = pd.read_csv(
                MODEL_COMPARISON_PATH
            )

        else:

            model_comparison = joblib.load(
                MODEL_COMPARISON_PATH
            )

    except Exception:

        model_comparison = None


# =========================================================
# 14. SAVE HISTORY
# =========================================================

def save_prediction_history(data):

    history_df = pd.DataFrame(
        [data]
    )

    if os.path.exists(
        HISTORY_PATH
    ):

        history_df.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        history_df.to_csv(
            HISTORY_PATH,
            mode="w",
            header=True,
            index=False
        )


# =========================================================
# 15. CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 16. SIDEBAR
# =========================================================

st.sidebar.title(
    "🎓 Student Predictor"
)

st.sidebar.write(
    "Machine Learning Based Academic "
    "Performance Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Prediction",
        "📊 Model Dashboard",
        "🏆 Model Comparison",
        "📜 Prediction History",
        "ℹ️ About Project"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🎓 Student Performance Predictor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Academic Performance Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header(
        "👋 Welcome"
    )

    st.write(
        "This application uses Machine Learning "
        "to predict a student's academic performance "
        "based on academic, behavioural and "
        "learning-related information."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "🤖 Machine Learning Prediction\n\n"
            "Predict student performance using "
            "student-related features."
        )

        st.info(
            "📊 Performance Analysis\n\n"
            "Analyze academic and learning indicators."
        )

        st.info(
            "💡 Personalized Recommendations\n\n"
            "Get suggestions based on student inputs."
        )

    with col2:

        st.info(
            "📈 Prediction Probability\n\n"
            "View probability distribution."
        )

        st.info(
            "🎯 Model Evaluation\n\n"
            "View accuracy, precision, recall and F1-score."
        )

        st.info(
            "📜 Prediction History\n\n"
            "Store and review previous predictions."
        )

    st.subheader(
        "🔄 Project Workflow"
    )

    st.write(
        """
        Student Input
        ↓
        Input Validation
        ↓
        Feature Engineering
        ↓
        Machine Learning Model
        ↓
        Performance Prediction
        ↓
        Analysis & Recommendations
        ↓
        Prediction History
        """
    )

    st.success(
        "👉 Go to the Prediction page from the sidebar."
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🤖 Prediction":

    st.markdown(
        '<div class="main-title">'
        '🤖 Student Performance Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter student information and predict academic performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------------------
    # ACADEMIC
    # -----------------------------------------------------

    st.header(
        "📚 Academic Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

    with col2:

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

    with col3:

        assignment_completion = st.number_input(
            "Assignment Completion (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    st.divider()

    # -----------------------------------------------------
    # STUDENT
    # -----------------------------------------------------

    st.header(
        "👤 Student Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=30,
            value=18,
            step=1
        )

    with col2:

        gender_label = st.selectbox(
            "Gender",
            [
                "Option 0",
                "Option 1"
            ]
        )

        gender = int(
            gender_label.split()[-1]
        )

    with col3:

        learning_style_label = st.selectbox(
            "Learning Style",
            [
                "Option 0",
                "Option 1",
                "Option 2",
                "Option 3"
            ]
        )

        learning_style = int(
            learning_style_label.split()[-1]
        )

    st.divider()

    # -----------------------------------------------------
    # RESOURCES
    # -----------------------------------------------------

    st.header(
        "💻 Learning & Resources"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        resources_label = st.selectbox(
            "Resources",
            [
                "Option 0",
                "Option 1",
                "Option 2"
            ]
        )

        resources = int(
            resources_label.split()[-1]
        )

    with col2:

        internet_label = st.selectbox(
            "Internet Access",
            [
                "Option 0",
                "Option 1"
            ]
        )

        internet = int(
            internet_label.split()[-1]
        )

    with col3:

        online_courses = st.number_input(
            "Online Courses",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

    st.divider()

    # -----------------------------------------------------
    # BEHAVIOUR
    # -----------------------------------------------------

    st.header(
        "📈 Behaviour & Engagement"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        motivation_label = st.selectbox(
            "Motivation",
            [
                "Option 0",
                "Option 1",
                "Option 2"
            ]
        )

        motivation = int(
            motivation_label.split()[-1]
        )

    with col2:

        discussions_label = st.selectbox(
            "Discussion Participation",
            [
                "Option 0",
                "Option 1"
            ]
        )

        discussions = int(
            discussions_label.split()[-1]
        )

    with col3:

        extracurricular_label = st.selectbox(
            "Extracurricular Activities",
            [
                "Option 0",
                "Option 1"
            ]
        )

        extracurricular = int(
            extracurricular_label.split()[-1]
        )

    st.divider()

    # -----------------------------------------------------
    # OTHER
    # -----------------------------------------------------

    st.header(
        "🧠 Other Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        edutech_label = st.selectbox(
            "EduTech Usage",
            [
                "Option 0",
                "Option 1"
            ]
        )

        edutech = int(
            edutech_label.split()[-1]
        )

    with col2:

        stress_level_label = st.selectbox(
            "Stress Level",
            [
                "Option 0",
                "Option 1",
                "Option 2"
            ]
        )

        stress_level = int(
            stress_level_label.split()[-1]
        )

    st.divider()

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    academic_engagement = (
        attendance +
        assignment_completion
    ) / 2


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Performance",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "StudyHours": [study_hours],

            "Attendance": [attendance],

            "Resources": [resources],

            "Extracurricular": [extracurricular],

            "Motivation": [motivation],

            "Internet": [internet],

            "Gender": [gender],

            "Age": [age],

            "LearningStyle": [learning_style],

            "OnlineCourses": [online_courses],

            "Discussions": [discussions],

            "AssignmentCompletion": [
                assignment_completion
            ],

            "EduTech": [edutech],

            "StressLevel": [stress_level],

            "AcademicEngagement": [
                academic_engagement
            ]
        })


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            prediction_input = input_data

            # First try direct prediction
            try:

                prediction = model.predict(
                    prediction_input
                )[0]

            except Exception as direct_error:

                # If separate preprocessor exists,
                # transform the input and try again

                if preprocessor is None:

                    raise direct_error

                transformed_data = (
                    preprocessor.transform(
                        input_data
                    )
                )

                prediction = model.predict(
                    transformed_data
                )[0]


        except Exception as e:

            st.error(
                "❌ MODEL PREDICTION FAILED"
            )

            st.write(
                "Model loaded from:"
            )

            st.code(
                MODEL_PATH
            )

            st.write(
                "Error:"
            )

            st.code(
                str(e)
            )

            st.stop()


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        prediction_probabilities = None

        try:

            if hasattr(
                model,
                "predict_proba"
            ):

                try:

                    prediction_probabilities = (
                        model.predict_proba(
                            input_data
                        )[0]
                    )

                except Exception:

                    if preprocessor is not None:

                        transformed_data = (
                            preprocessor.transform(
                                input_data
                            )
                        )

                        prediction_probabilities = (
                            model.predict_proba(
                                transformed_data
                            )[0]
                        )

        except Exception:

            prediction_probabilities = None


        # -------------------------------------------------
        # LABELS
        # -------------------------------------------------

        grade_labels = {

            0: "Low Performance",

            1: "Below Average Performance",

            2: "Good Performance",

            3: "Excellent Performance"
        }


        try:

            prediction_int = int(
                prediction
            )

        except Exception:

            prediction_int = prediction


        result = grade_labels.get(
            prediction_int,
            f"Performance Class {prediction}"
        )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence_percentage = 0

        if prediction_probabilities is not None:

            try:

                classes = getattr(
                    model,
                    "classes_",
                    list(
                        range(
                            len(
                                prediction_probabilities
                            )
                        )
                    )
                )

                if prediction in classes:

                    prediction_index = list(
                        classes
                    ).index(
                        prediction
                    )

                    confidence_percentage = (
                        prediction_probabilities[
                            prediction_index
                        ] * 100
                    )

            except Exception:

                confidence_percentage = (
                    max(
                        prediction_probabilities
                    ) * 100
                )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header(
            "🎯 Prediction Result"
        )

        st.success(
            f"Predicted Performance: {result}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence_percentage:.2f}%"
        )


        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        recommendations = []


        if study_hours < 3:

            recommendations.append(
                "📚 Try to increase your study time gradually."
            )


        if attendance < 75:

            recommendations.append(
                "🏫 Try to attend classes more regularly."
            )


        if assignment_completion < 70:

            recommendations.append(
                "📝 Complete more assignments on time."
            )


        if online_courses == 0:

            recommendations.append(
                "💻 Consider taking an online course."
            )


        if motivation == 0:

            recommendations.append(
                "🎯 Try setting small daily academic goals."
            )


        if discussions == 0:

            recommendations.append(
                "🙋 Participate more actively in discussions."
            )


        if extracurricular == 0:

            recommendations.append(
                "⚽ Consider suitable extracurricular activities."
            )


        if stress_level == 2:

            recommendations.append(
                "🧘 Maintain a balanced study routine."
            )


        if len(recommendations) == 0:

            recommendations.append(
                "🌟 Your current inputs look balanced. "
                "Keep maintaining your study routine!"
            )


        st.subheader(
            "💡 Personalized Recommendations"
        )


        for recommendation in recommendations:

            st.info(
                recommendation
            )


        # -------------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------------

        history_record = {

            "Date Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "Age": age,

            "Gender Code": gender,

            "Learning Style Code":
                learning_style,

            "Study Hours":
                study_hours,

            "Attendance (%)":
                attendance,

            "Assignment Completion (%)":
                assignment_completion,

            "Resources Code":
                resources,

            "Internet Access Code":
                internet,

            "Online Courses":
                online_courses,

            "Motivation Code":
                motivation,

            "Discussion Participation Code":
                discussions,

            "Extracurricular Activities Code":
                extracurricular,

            "EduTech Usage Code":
                edutech,

            "Stress Level Code":
                stress_level,

            "Academic Engagement":
                round(
                    academic_engagement,
                    2
                ),

            "Predicted Performance":
                result,

            "Confidence (%)":
                round(
                    confidence_percentage,
                    2
                )
        }


        save_prediction_history(
            history_record
        )


        st.success(
            "✅ Prediction saved to Prediction History."
        )


        # -------------------------------------------------
        # STUDENT DASHBOARD
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Student Analysis Dashboard"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Age",
                f"{age} years"
            )

        with col2:

            st.metric(
                "Study Hours",
                f"{study_hours} hrs/day"
            )

        with col3:

            st.metric(
                "Attendance",
                f"{attendance:.0f}%"
            )


        # -------------------------------------------------
        # ACADEMIC CHART
        # -------------------------------------------------

        academic_data = pd.DataFrame({

            "Indicator": [
                "Attendance",
                "Assignment Completion",
                "Academic Engagement"
            ],

            "Score": [
                attendance,
                assignment_completion,
                academic_engagement
            ]
        })


        st.subheader(
            "📚 Academic Indicators"
        )

        st.bar_chart(
            academic_data.set_index(
                "Indicator"
            )
        )


        # -------------------------------------------------
        # LEARNING CHART
        # -------------------------------------------------

        learning_data = pd.DataFrame({

            "Activity": [
                "Study Hours",
                "Online Courses"
            ],

            "Value": [
                study_hours,
                online_courses
            ]
        })


        st.subheader(
            "💻 Learning Activity"
        )

        st.bar_chart(
            learning_data.set_index(
                "Activity"
            )
        )


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if prediction_probabilities is not None:

            st.subheader(
                "📊 Prediction Probability"
            )


            probability_data = pd.DataFrame({

                "Performance": [
                    grade_labels.get(
                        i,
                        f"Class {i}"
                    )
                    for i in range(
                        len(
                            prediction_probabilities
                        )
                    )
                ],

                "Probability": [
                    round(
                        p * 100,
                        2
                    )
                    for p in prediction_probabilities
                ]
            })


            st.dataframe(
                probability_data,
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "📈 Performance Probability Chart"
            )


            st.bar_chart(
                probability_data.set_index(
                    "Performance"
                )
            )


        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        report = pd.DataFrame({

            "Parameter": [
                "Age",
                "Gender Code",
                "Learning Style Code",
                "Study Hours",
                "Attendance (%)",
                "Assignment Completion (%)",
                "Resources Code",
                "Internet Access Code",
                "Online Courses",
                "Motivation Code",
                "Discussion Participation Code",
                "Extracurricular Activities Code",
                "EduTech Usage Code",
                "Stress Level Code",
                "Academic Engagement",
                "Predicted Performance",
                "Prediction Confidence (%)"
            ],

            "Value": [
                age,
                gender,
                learning_style,
                study_hours,
                attendance,
                assignment_completion,
                resources,
                internet,
                online_courses,
                motivation,
                discussions,
                extracurricular,
                edutech,
                stress_level,
                round(
                    academic_engagement,
                    2
                ),
                result,
                round(
                    confidence_percentage,
                    2
                )
            ]
        })


        st.subheader(
            "📥 Download Student Report"
        )


        st.download_button(
            "📥 Download Student Report",
            report.to_csv(
                index=False
            ),
            "student_performance_report.csv",
            "text/csv",
            use_container_width=True
        )


# =========================================================
# MODEL DASHBOARD
# =========================================================

elif page == "📊 Model Dashboard":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Performance Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if len(metrics) == 0:

        st.warning(
            "⚠️ Model metrics file is not available."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            if "accuracy" in metrics:

                st.metric(
                    "Accuracy",
                    f"{metrics['accuracy'] * 100:.2f}%"
                )

        with col2:

            if "precision" in metrics:

                st.metric(
                    "Precision",
                    f"{metrics['precision'] * 100:.2f}%"
                )

        with col3:

            if "recall" in metrics:

                st.metric(
                    "Recall",
                    f"{metrics['recall'] * 100:.2f}%"
                )

        with col4:

            if "f1_score" in metrics:

                st.metric(
                    "F1 Score",
                    f"{metrics['f1_score'] * 100:.2f}%"
                )


    if cm is not None:

        st.divider()

        st.subheader(
            "📊 Confusion Matrix"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.imshow(cm)

        ax.set_title(
            "Model Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        ax.set_xticks(
            range(len(cm))
        )

        ax.set_yticks(
            range(len(cm))
        )


        for i in range(len(cm)):

            for j in range(len(cm)):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )


        plt.tight_layout()

        st.pyplot(
            fig
        )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🏆 Model Comparison":

    st.markdown(
        '<div class="main-title">'
        '🏆 Machine Learning Model Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if model_comparison is None:

        st.warning(
            "⚠️ Model comparison file not found."
        )

    else:

        required_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        missing = [
            c
            for c in required_columns
            if c not in model_comparison.columns
        ]

        if len(missing) > 0:

            st.error(
                "❌ Missing columns:"
            )

            st.write(
                missing
            )

        else:

            display = model_comparison.copy()

            display["Accuracy"] = (
                display["Accuracy"] * 100
            ).round(2)

            display["Precision"] = (
                display["Precision"] * 100
            ).round(2)

            display["Recall"] = (
                display["Recall"] * 100
            ).round(2)

            display["F1 Score"] = (
                display["F1 Score"] * 100
            ).round(2)


            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )


            best = model_comparison.loc[
                model_comparison[
                    "F1 Score"
                ].idxmax()
            ]


            st.success(
                f"🏆 Best Performing Model: "
                f"{best['Model']}"
            )


            chart = display[
                [
                    "Model",
                    "Accuracy"
                ]
            ].set_index(
                "Model"
            )


            st.subheader(
                "📈 Accuracy Comparison"
            )

            st.bar_chart(
                chart
            )


# =========================================================
# HISTORY
# =========================================================

elif page == "📜 Prediction History":

    st.markdown(
        '<div class="main-title">'
        '📜 Prediction History'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if not os.path.exists(
        HISTORY_PATH
    ):

        st.info(
            "ℹ️ No prediction history available yet."
        )

    else:

        try:

            history_df = pd.read_csv(
                HISTORY_PATH
            )

            if history_df.empty:

                st.info(
                    "ℹ️ No predictions recorded yet."
                )

            else:

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Predictions",
                        len(history_df)
                    )

                with col2:

                    st.metric(
                        "Latest Prediction",
                        history_df.iloc[-1][
                            "Predicted Performance"
                        ]
                    )

                with col3:

                    st.metric(
                        "Average Confidence",
                        f"{history_df['Confidence (%)'].mean():.2f}%"
                    )


                st.divider()

                st.subheader(
                    "📋 Previous Predictions"
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )


                st.subheader(
                    "📈 Performance Distribution"
                )

                st.bar_chart(
                    history_df[
                        "Predicted Performance"
                    ].value_counts()
                )


                st.subheader(
                    "🎯 Confidence History"
                )

                st.line_chart(
                    history_df[
                        [
                            "Date Time",
                            "Confidence (%)"
                        ]
                    ].set_index(
                        "Date Time"
                    )
                )


                st.download_button(
                    "📥 Download Complete History",
                    history_df.to_csv(
                        index=False
                    ),
                    "prediction_history.csv",
                    "text/csv",
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "❌ Error loading prediction history."
            )

            st.code(
                str(e)
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About Student Performance Predictor'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header(
        "🎯 Problem Statement"
    )

    st.write(
        "The objective is to develop a Machine Learning "
        "system that predicts student academic performance "
        "using academic, behavioural and learning-related "
        "features."
    )


    st.header(
        "🛠️ Technologies Used"
    )

    technologies = pd.DataFrame({

        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "Joblib",
            "Streamlit",
            "Matplotlib"
        ],

        "Purpose": [
            "Programming Language",
            "Data Processing",
            "Machine Learning",
            "Model Saving",
            "Web Application",
            "Visualization"
        ]
    })


    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )


    st.header(
        "🔄 Machine Learning Workflow"
    )

    st.write(
        """
        1. Dataset Collection
        2. Data Cleaning
        3. Exploratory Data Analysis
        4. Feature Engineering
        5. Data Preprocessing
        6. Model Training
        7. Model Evaluation
        8. Model Saving
        9. Streamlit Application
        10. Prediction
        11. Model Comparison
        12. Recommendations
        13. Prediction History
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Student Performance Predictor • "
    "End-to-End Machine Learning Project"
)