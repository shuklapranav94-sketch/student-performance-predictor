import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# 1. BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# 2. DIRECTORIES
# =========================================================

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

# Automatically create data folder if it does not exist
os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# =========================================================
# 3. FILE PATHS
# =========================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "student_performance_final_model.pkl"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.pkl"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODELS_DIR,
    "confusion_matrix.pkl"
)

MODEL_COMPARISON_PATH = os.path.join(
    MODELS_DIR,
    "model_comparison_results.csv"
)

HISTORY_PATH = os.path.join(
    DATA_DIR,
    "prediction_history.csv"
)


# =========================================================
# 4. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# 5. LOAD MODEL
# =========================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Model file not found. "
        "Please check the models folder."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# =========================================================
# 6. LOAD MODEL METRICS
# =========================================================

try:

    metrics = joblib.load(
        METRICS_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Model metrics file not found."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model metrics: {e}"
    )

    st.stop()


# =========================================================
# 7. LOAD CONFUSION MATRIX
# =========================================================

try:

    cm = joblib.load(
        CONFUSION_MATRIX_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Confusion matrix file not found."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading confusion matrix: {e}"
    )

    st.stop()


# =========================================================
# 8. LOAD MODEL COMPARISON
# =========================================================

try:

    model_comparison = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Model comparison file not found."
    )

    st.info(
        "Please make sure "
        "'model_comparison_results.csv' "
        "exists inside the models folder."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model comparison: {e}"
    )

    st.stop()


# =========================================================
# 9. SAVE PREDICTION HISTORY FUNCTION
# =========================================================

def save_prediction_history(data):

    history_df = pd.DataFrame(
        [data]
    )

    if os.path.exists(HISTORY_PATH):

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
# 10. CUSTOM CSS
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
# 11. SIDEBAR NAVIGATION
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
# HOME PAGE
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
        "This application uses Machine Learning to "
        "predict a student's academic performance "
        "based on academic, behavioural and "
        "learning-related information."
    )

    st.subheader(
        "🚀 Project Features"
    )

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.info(
            "🤖 Machine Learning Prediction\n\n"
            "Predict student performance using "
            "multiple student-related features."
        )

        st.info(
            "📊 Performance Analysis\n\n"
            "Analyze academic and learning indicators "
            "using interactive charts."
        )

        st.info(
            "💡 Personalized Recommendations\n\n"
            "Get suggestions based on the student's "
            "selected inputs."
        )

    with feature_col2:

        st.info(
            "📈 Prediction Probability\n\n"
            "View probability distribution for "
            "different performance classes."
        )

        st.info(
            "🎯 Model Evaluation\n\n"
            "View accuracy, precision, recall, "
            "F1-score and confusion matrix."
        )

        st.info(
            "📥 Download Report\n\n"
            "Download the student's prediction report "
            "as a CSV file."
        )

        st.info(
            "📜 Prediction History\n\n"
            "Store and review previous student predictions."
        )

    st.subheader(
        "🔄 Project Workflow"
    )

    st.write(
        """
        **Student Input**
        ↓
        **Input Validation**
        ↓
        **Feature Engineering**
        ↓
        **Machine Learning Model**
        ↓
        **Performance Prediction**
        ↓
        **Analysis & Recommendations**
        ↓
        **Prediction History**
        """
    )

    st.success(
        "👉 Go to the Prediction page from the "
        "sidebar to test the model."
    )


# =========================================================
# PREDICTION PAGE
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


    # =====================================================
    # ACADEMIC INFORMATION
    # =====================================================

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


    # =====================================================
    # STUDENT INFORMATION
    # =====================================================

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


    # =====================================================
    # LEARNING & RESOURCES
    # =====================================================

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


    # =====================================================
    # BEHAVIOUR & ENGAGEMENT
    # =====================================================

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


    # =====================================================
    # OTHER INFORMATION
    # =====================================================

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


    # =====================================================
    # STEP 23
    # ADVANCED INPUT VALIDATION
    # =====================================================

    validation_errors = []


    # =====================================================
    # NUMERIC VALIDATION
    # =====================================================

    if pd.isna(study_hours):

        validation_errors.append(
            "Study hours cannot be empty."
        )


    if pd.isna(attendance):

        validation_errors.append(
            "Attendance cannot be empty."
        )


    if pd.isna(assignment_completion):

        validation_errors.append(
            "Assignment completion cannot be empty."
        )


    if pd.isna(age):

        validation_errors.append(
            "Age cannot be empty."
        )


    if pd.isna(online_courses):

        validation_errors.append(
            "Online courses cannot be empty."
        )


    # =====================================================
    # RANGE VALIDATION
    # =====================================================

    if not pd.isna(study_hours):

        if study_hours < 0 or study_hours > 24:

            validation_errors.append(
                "❌ Study hours must be between 0 and 24."
            )


    if not pd.isna(attendance):

        if attendance < 0 or attendance > 100:

            validation_errors.append(
                "❌ Attendance must be between 0% and 100%."
            )


    if not pd.isna(assignment_completion):

        if assignment_completion < 0 or assignment_completion > 100:

            validation_errors.append(
                "❌ Assignment completion must be between 0% and 100%."
            )


    if not pd.isna(age):

        if age < 10 or age > 30:

            validation_errors.append(
                "❌ Age must be between 10 and 30."
            )


    if not pd.isna(online_courses):

        if online_courses < 0 or online_courses > 20:

            validation_errors.append(
                "❌ Online courses must be between 0 and 20."
            )


    # =====================================================
    # CATEGORY VALIDATION
    # =====================================================

    if gender not in [0, 1]:

        validation_errors.append(
            "❌ Invalid gender value selected."
        )


    if learning_style not in [0, 1, 2, 3]:

        validation_errors.append(
            "❌ Invalid learning style selected."
        )


    if resources not in [0, 1, 2]:

        validation_errors.append(
            "❌ Invalid resources value selected."
        )


    if internet not in [0, 1]:

        validation_errors.append(
            "❌ Invalid internet access value selected."
        )


    if motivation not in [0, 1, 2]:

        validation_errors.append(
            "❌ Invalid motivation value selected."
        )


    if discussions not in [0, 1]:

        validation_errors.append(
            "❌ Invalid discussion value selected."
        )


    if extracurricular not in [0, 1]:

        validation_errors.append(
            "❌ Invalid extracurricular value selected."
        )


    if edutech not in [0, 1]:

        validation_errors.append(
            "❌ Invalid EduTech value selected."
        )


    if stress_level not in [0, 1, 2]:

        validation_errors.append(
            "❌ Invalid stress level selected."
        )


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    academic_engagement = (
        attendance + assignment_completion
    ) / 2


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.write("")

    predict_button = st.button(
        "🔮 Predict Performance",
        use_container_width=True
    )


    if predict_button:

        # =================================================
        # DISPLAY VALIDATION ERRORS
        # =================================================

        if len(validation_errors) > 0:

            st.error(
                "❌ Please correct the following inputs:"
            )

            for error in validation_errors:

                st.warning(
                    error
                )


        else:

            try:

                # =========================================
                # CREATE INPUT DATAFRAME
                # =========================================

                input_data = pd.DataFrame({

                    "StudyHours": [study_hours],

                    "Attendance": [attendance],

                    "Resources": [resources],

                    "Extracurricular": [
                        extracurricular
                    ],

                    "Motivation": [motivation],

                    "Internet": [internet],

                    "Gender": [gender],

                    "Age": [age],

                    "LearningStyle": [
                        learning_style
                    ],

                    "OnlineCourses": [
                        online_courses
                    ],

                    "Discussions": [
                        discussions
                    ],

                    "AssignmentCompletion": [
                        assignment_completion
                    ],

                    "EduTech": [edutech],

                    "StressLevel": [
                        stress_level
                    ],

                    "AcademicEngagement": [
                        academic_engagement
                    ]
                })


                # =========================================
                # VERIFY REQUIRED FEATURES
                # =========================================

                expected_features = [
                    "StudyHours",
                    "Attendance",
                    "Resources",
                    "Extracurricular",
                    "Motivation",
                    "Internet",
                    "Gender",
                    "Age",
                    "LearningStyle",
                    "OnlineCourses",
                    "Discussions",
                    "AssignmentCompletion",
                    "EduTech",
                    "StressLevel",
                    "AcademicEngagement"
                ]


                missing_features = [
                    feature
                    for feature in expected_features
                    if feature not in input_data.columns
                ]


                if len(missing_features) > 0:

                    st.error(
                        "❌ Some required model features are missing."
                    )

                    st.write(
                        "Missing features:"
                    )

                    st.write(
                        missing_features
                    )

                    st.stop()


                # =========================================
                # SAFE MODEL PREDICTION
                # =========================================

                try:

                    prediction = model.predict(
                        input_data
                    )[0]

                except Exception as prediction_error:

                    st.error(
                        "❌ Model prediction failed."
                    )

                    with st.expander(
                        "Technical Error"
                    ):

                        st.code(
                            str(prediction_error)
                        )

                    st.stop()


                # =========================================
                # SAFE PREDICTION PROBABILITY
                # =========================================

                try:

                    if not hasattr(
                        model,
                        "predict_proba"
                    ):

                        st.warning(
                            "⚠️ This model does not support "
                            "prediction probabilities."
                        )

                        prediction_probabilities = None

                    else:

                        prediction_probabilities = (
                            model.predict_proba(
                                input_data
                            )[0]
                        )

                except Exception as probability_error:

                    st.warning(
                        "⚠️ Prediction probability could "
                        "not be calculated."
                    )

                    with st.expander(
                        "Technical Error"
                    ):

                        st.code(
                            str(probability_error)
                        )

                    prediction_probabilities = None


                # =========================================
                # PREDICTION LABELS
                # =========================================

                grade_labels = {

                    0: "Low Performance",

                    1: "Below Average Performance",

                    2: "Good Performance",

                    3: "Excellent Performance"
                }


                result = grade_labels.get(
                    prediction,
                    "Unknown"
                )


                # =========================================
                # CONFIDENCE
                # =========================================

                if prediction_probabilities is not None:

                    if prediction < len(
                        prediction_probabilities
                    ):

                        confidence = (
                            prediction_probabilities[
                                prediction
                            ]
                        )

                        confidence_percentage = (
                            confidence * 100
                        )

                    else:

                        confidence_percentage = 0

                else:

                    confidence_percentage = 0


                # =========================================
                # RECOMMENDATIONS
                # =========================================

                recommendations = []


                if study_hours < 3:

                    recommendations.append(
                        "📚 Try to increase your study time gradually. "
                        "Aim for at least 3–4 focused study hours."
                    )


                if attendance < 75:

                    recommendations.append(
                        "🏫 Your attendance is relatively low. "
                        "Try to attend classes more regularly."
                    )


                if assignment_completion < 70:

                    recommendations.append(
                        "📝 Complete more assignments on time "
                        "to improve your academic performance."
                    )


                if online_courses == 0:

                    recommendations.append(
                        "💻 Consider taking an online course "
                        "related to your subjects."
                    )


                if motivation == 0:

                    recommendations.append(
                        "🎯 Try setting small daily academic goals "
                        "to improve your motivation."
                    )


                if discussions == 0:

                    recommendations.append(
                        "🙋 Participate more actively in class "
                        "discussions and academic activities."
                    )


                if extracurricular == 0:

                    recommendations.append(
                        "⚽ Consider participating in suitable "
                        "extracurricular activities for a balanced routine."
                    )


                if stress_level == 2:

                    recommendations.append(
                        "🧘 Your selected stress level is high. "
                        "Take regular breaks and maintain a balanced study routine."
                    )


                if internet == 0:

                    recommendations.append(
                        "🌐 Access to reliable learning resources "
                        "can help with your studies."
                    )


                if len(recommendations) == 0:

                    recommendations.append(
                        "🌟 Your current inputs look balanced. "
                        "Keep maintaining your study routine!"
                    )


                # =========================================
                # DISPLAY RESULT
                # =========================================

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


                # =========================================
                # STEP 25
                # SAVE PREDICTION HISTORY
                # =========================================

                history_record = {

                    "Date Time":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Age":
                        age,

                    "Gender Code":
                        gender,

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


                # =========================================
                # TECHNICAL DETAILS
                # =========================================

                with st.expander(
                    "🔍 Technical Prediction Details"
                ):

                    st.write(
                        f"Predicted Class: {prediction}"
                    )


                # =========================================
                # RECOMMENDATIONS
                # =========================================

                st.subheader(
                    "💡 Personalized Recommendations"
                )


                for recommendation in recommendations:

                    st.info(
                        recommendation
                    )


                # =========================================
                # STUDENT ANALYSIS
                # =========================================

                st.divider()

                st.subheader(
                    "📊 Student Analysis Dashboard"
                )


                st.write(
                    "### 👤 Student Profile"
                )


                profile_col1, profile_col2, profile_col3 = st.columns(3)


                with profile_col1:

                    st.metric(
                        "Age",
                        f"{age} years"
                    )


                with profile_col2:

                    st.metric(
                        "Study Hours",
                        f"{study_hours} hrs/day"
                    )


                with profile_col3:

                    st.metric(
                        "Attendance",
                        f"{attendance:.0f}%"
                    )


                # =========================================
                # ACADEMIC INDICATORS
                # =========================================

                st.write(
                    "### 📚 Academic Indicators"
                )


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


                st.bar_chart(
                    academic_data.set_index(
                        "Indicator"
                    )
                )


                # =========================================
                # LEARNING ACTIVITY
                # =========================================

                st.write(
                    "### 💻 Learning Activity"
                )


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


                st.bar_chart(
                    learning_data.set_index(
                        "Activity"
                    )
                )


                # =========================================
                # ENGAGEMENT SUMMARY
                # =========================================

                st.write(
                    "### 📈 Engagement Summary"
                )


                engagement_col1, engagement_col2, engagement_col3 = st.columns(3)


                with engagement_col1:

                    st.metric(
                        "Motivation Level",
                        str(motivation)
                    )


                with engagement_col2:

                    st.metric(
                        "Discussion Level",
                        str(discussions)
                    )


                with engagement_col3:

                    st.metric(
                        "Academic Engagement",
                        f"{academic_engagement:.2f}"
                    )


                # =========================================
                # PROBABILITY TABLE
                # =========================================

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
                                probability * 100,
                                2
                            )
                            for probability in (
                                prediction_probabilities
                            )
                        ]
                    })


                    st.dataframe(
                        probability_data,
                        use_container_width=True,
                        hide_index=True
                    )


                    # =====================================
                    # PROBABILITY CHART
                    # =====================================

                    st.subheader(
                        "📈 Performance Probability Chart"
                    )


                    chart_data = (
                        probability_data
                        .set_index("Performance")
                    )


                    st.bar_chart(
                        chart_data["Probability"]
                    )

                else:

                    st.info(
                        "ℹ️ Probability information is "
                        "not available for this model."
                    )


                # =========================================
                # DOWNLOAD REPORT
                # =========================================

                st.divider()

                st.subheader(
                    "📥 Download Student Report"
                )


                report_data = {

                    "Student Performance Report": "",

                    "Report Type":
                        "Machine Learning Prediction Report",

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

                    "Prediction Confidence (%)":
                        round(
                            confidence_percentage,
                            2
                        )
                }


                report_df = pd.DataFrame(
                    list(
                        report_data.items()
                    ),
                    columns=[
                        "Parameter",
                        "Value"
                    ]
                )


                recommendation_rows = []


                for index, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    recommendation_rows.append({

                        "Parameter":
                            f"Recommendation {index}",

                        "Value":
                            recommendation
                    })


                recommendation_df = pd.DataFrame(
                    recommendation_rows
                )


                final_report = pd.concat(
                    [
                        report_df,
                        recommendation_df
                    ],
                    ignore_index=True
                )


                report_csv = final_report.to_csv(
                    index=False
                )


                st.download_button(

                    label="📥 Download Student Report",

                    data=report_csv,

                    file_name="student_performance_report.csv",

                    mime="text/csv",

                    use_container_width=True
                )


            # =============================================
            # ERROR HANDLING
            # =============================================

            except ValueError as e:

                st.error(
                    "❌ Invalid input format. "
                    "Please check the entered values."
                )

                with st.expander(
                    "Technical Error"
                ):

                    st.code(
                        str(e)
                    )


            except Exception as e:

                st.error(
                    "❌ Something went wrong while "
                    "making the prediction."
                )

                with st.expander(
                    "Technical Error"
                ):

                    st.code(
                        str(e)
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

    st.markdown(
        '<div class="subtitle">'
        'Evaluation of the trained Machine Learning model'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    st.subheader(
        "📈 Model Evaluation Metrics"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )


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


    st.divider()


    st.subheader(
        "🤖 Model Information"
    )


    st.write(
        "The model predicts student academic "
        "performance using academic, behavioural "
        "and learning-related features."
    )


    st.write(
        "The performance metrics above are calculated "
        "using the test dataset."
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

    st.markdown(
        '<div class="subtitle">'
        'Comparison of different Machine Learning algorithms'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in model_comparison.columns
    ]


    if len(missing_columns) > 0:

        st.error(
            "❌ Required columns are missing from "
            "model_comparison_results.csv"
        )

        st.write(
            "Missing columns:"
        )

        st.write(
            missing_columns
        )

    else:

        st.subheader(
            "📊 Model Performance Comparison"
        )


        comparison_display = (
            model_comparison.copy()
        )


        comparison_display["Accuracy"] = (
            comparison_display["Accuracy"] * 100
        ).round(2)


        comparison_display["Precision"] = (
            comparison_display["Precision"] * 100
        ).round(2)


        comparison_display["Recall"] = (
            comparison_display["Recall"] * 100
        ).round(2)


        comparison_display["F1 Score"] = (
            comparison_display["F1 Score"] * 100
        ).round(2)


        comparison_display = (
            comparison_display.rename(
                columns={
                    "Accuracy": "Accuracy (%)",
                    "Precision": "Precision (%)",
                    "Recall": "Recall (%)",
                    "F1 Score": "F1 Score (%)"
                }
            )
        )


        st.dataframe(
            comparison_display,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # BEST MODEL
        # =================================================

        best_model_row = model_comparison.loc[
            model_comparison["F1 Score"].idxmax()
        ]


        best_model_name = (
            best_model_row["Model"]
        )


        best_model_accuracy = (
            best_model_row["Accuracy"] * 100
        )


        best_model_f1 = (
            best_model_row["F1 Score"] * 100
        )


        st.success(
            f"🏆 Best Performing Model: "
            f"{best_model_name}"
        )


        best_col1, best_col2 = st.columns(2)


        with best_col1:

            st.metric(
                "Best Model Accuracy",
                f"{best_model_accuracy:.2f}%"
            )


        with best_col2:

            st.metric(
                "Best Model F1 Score",
                f"{best_model_f1:.2f}%"
            )


        # =================================================
        # ACCURACY
        # =================================================

        st.divider()

        st.subheader(
            "📈 Accuracy Comparison"
        )


        accuracy_chart = (
            comparison_display[
                [
                    "Model",
                    "Accuracy (%)"
                ]
            ]
            .set_index("Model")
        )


        st.bar_chart(
            accuracy_chart
        )


        # =================================================
        # PRECISION
        # =================================================

        st.subheader(
            "🎯 Precision Comparison"
        )


        precision_chart = (
            comparison_display[
                [
                    "Model",
                    "Precision (%)"
                ]
            ]
            .set_index("Model")
        )


        st.bar_chart(
            precision_chart
        )


        # =================================================
        # RECALL
        # =================================================

        st.subheader(
            "🔍 Recall Comparison"
        )


        recall_chart = (
            comparison_display[
                [
                    "Model",
                    "Recall (%)"
                ]
            ]
            .set_index("Model")
        )


        st.bar_chart(
            recall_chart
        )


        # =================================================
        # F1 SCORE
        # =================================================

        st.subheader(
            "⭐ F1 Score Comparison"
        )


        f1_chart = (
            comparison_display[
                [
                    "Model",
                    "F1 Score (%)"
                ]
            ]
            .set_index("Model")
        )


        st.bar_chart(
            f1_chart
        )


        # =================================================
        # COMPLETE METRICS
        # =================================================

        st.subheader(
            "📊 Complete Metrics Comparison"
        )


        all_metrics_chart = (
            comparison_display[
                [
                    "Model",
                    "Accuracy (%)",
                    "Precision (%)",
                    "Recall (%)",
                    "F1 Score (%)"
                ]
            ]
            .set_index("Model")
        )


        st.bar_chart(
            all_metrics_chart
        )


        # =================================================
        # MODEL SELECTION
        # =================================================

        st.divider()

        st.subheader(
            "🧠 Why Was This Model Selected?"
        )


        st.write(
            f"The **{best_model_name}** model was selected "
            f"as the final model because it achieved the "
            f"highest F1 Score among the evaluated models."
        )


        st.info(
            "F1 Score provides a balance between Precision "
            "and Recall and is useful when evaluating "
            "classification models."
        )


# =========================================================
# STEP 25
# PREDICTION HISTORY
# =========================================================

elif page == "📜 Prediction History":

    st.markdown(
        '<div class="main-title">'
        '📜 Prediction History'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'View previously generated student performance predictions'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =====================================================
    # CHECK HISTORY FILE
    # =====================================================

    if not os.path.exists(HISTORY_PATH):

        st.info(
            "ℹ️ No prediction history available yet."
        )

        st.write(
            "Go to the 🤖 Prediction page and "
            "make your first prediction."
        )


    else:

        try:

            history_df = pd.read_csv(
                HISTORY_PATH
            )


            # =================================================
            # EMPTY HISTORY CHECK
            # =================================================

            if history_df.empty:

                st.info(
                    "ℹ️ No predictions have been recorded yet."
                )

                st.write(
                    "Go to the 🤖 Prediction page and "
                    "make your first prediction."
                )


            else:

                # =============================================
                # HISTORY SUMMARY
                # =============================================

                st.subheader(
                    "📊 History Summary"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Total Predictions",
                        len(history_df)
                    )


                with col2:

                    latest_prediction = (
                        history_df.iloc[-1][
                            "Predicted Performance"
                        ]
                    )

                    st.metric(
                        "Latest Prediction",
                        latest_prediction
                    )


                with col3:

                    average_confidence = (
                        history_df[
                            "Confidence (%)"
                        ].mean()
                    )

                    st.metric(
                        "Average Confidence",
                        f"{average_confidence:.2f}%"
                    )


                st.divider()


                # =============================================
                # HISTORY TABLE
                # =============================================

                st.subheader(
                    "📋 Previous Predictions"
                )


                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )


                st.divider()


                # =============================================
                # PERFORMANCE DISTRIBUTION
                # =============================================

                st.subheader(
                    "📈 Performance Distribution"
                )


                performance_counts = (
                    history_df[
                        "Predicted Performance"
                    ]
                    .value_counts()
                )


                st.bar_chart(
                    performance_counts
                )


                st.divider()


                # =============================================
                # CONFIDENCE HISTORY
                # =============================================

                st.subheader(
                    "🎯 Prediction Confidence History"
                )


                confidence_chart = (
                    history_df[
                        [
                            "Date Time",
                            "Confidence (%)"
                        ]
                    ]
                    .set_index("Date Time")
                )


                st.line_chart(
                    confidence_chart
                )


                st.divider()


                # =============================================
                # DOWNLOAD HISTORY
                # =============================================

                st.subheader(
                    "📥 Download Prediction History"
                )


                history_csv = history_df.to_csv(
                    index=False
                )


                st.download_button(

                    label="📥 Download Complete History",

                    data=history_csv,

                    file_name="prediction_history.csv",

                    mime="text/csv",

                    use_container_width=True
                )


                st.divider()


                # =============================================
                # CLEAR HISTORY
                # =============================================

                st.subheader(
                    "🗑️ History Management"
                )


                st.warning(
                    "⚠️ Clearing history will permanently "
                    "remove all saved prediction records."
                )


                if st.button(
                    "🗑️ Clear Prediction History",
                    use_container_width=True
                ):

                    try:

                        os.remove(
                            HISTORY_PATH
                        )

                        st.success(
                            "✅ Prediction history cleared successfully."
                        )

                        st.rerun()

                    except Exception as clear_error:

                        st.error(
                            "❌ Could not clear prediction history."
                        )

                        with st.expander(
                            "Technical Error"
                        ):

                            st.code(
                                str(clear_error)
                            )


        except Exception as history_error:

            st.error(
                "❌ Error loading prediction history."
            )

            with st.expander(
                "Technical Error"
            ):

                st.code(
                    str(history_error)
                )


# =========================================================
# ABOUT PROJECT
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
        "The objective of this project is to develop "
        "a Machine Learning based system that can "
        "predict a student's academic performance "
        "using different academic, behavioural and "
        "learning-related features."
    )


    st.header(
        "🚀 Project Objective"
    )

    st.write(
        "The system aims to provide an early indication "
        "of student performance and provide useful "
        "recommendations based on the student's inputs."
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
            "Model Saving & Loading",
            "Web Application",
            "Data Visualization"
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

        10. Student Performance Prediction

        11. Model Comparison

        12. Personalized Recommendations

        13. Input Validation

        14. Error Handling

        15. Prediction History
        """
    )


    st.header(
        "📋 Project Features"
    )


    st.write(
        """
        • Student performance prediction

        • Prediction confidence

        • Performance probability

        • Personalized recommendations

        • Student analysis dashboard

        • Model performance dashboard

        • Confusion matrix

        • Downloadable prediction report

        • Input validation

        • Error handling

        • Machine Learning model comparison

        • Prediction history

        • Performance distribution

        • Confidence history

        • Downloadable prediction history
        """
    )


    st.success(
        "Student Performance Predictor is an "
        "end-to-end Machine Learning project."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Student Performance Predictor • "
    "End-to-End Machine Learning Project"
)