import re

COLUMN_MAPPING_INVERSED = {    
    # Publications
    'publication_authors': 'Authors',
    'publication_year': 'Publication year',
    'publication_title': 'Publication title',
    'publication_pmid': 'Publication PMID',
    'publication_doi': 'Publication DOI',
    'publication_journal': 'Publication journal',
    
    # Data
    'data_source': 'Source of data (e.g., EHR, registry, clinical trial, bespoke cohort)',
    'data_type': 'Data type (e.g., structured, unstructured text, imaging, time-series, genomics)',
    'data_available': 'Is data publicly available? (Yes, no, on request)',
    'data_sites_region': 'Study sites (region)',
    'data_sites_number': 'Study sites (number of centers)',
    'data_inclusion_criteria': 'Inclusion criteria',
    'data_exclusion_criteria': 'Exclusion criteria',
    'data_ethics_approval': 'Ethical approval',
    'data_consent_obtaiend': 'Consent',
    'data_study_settings': 'Study settings (e.g., inpatient, outpatient, ICU)',
    'data_recruitment_method': 'Recruitment method',
    'data_recruitment_dates': 'Recruitment dates',
    'data_collection_dates': 'Data collection dates',
    'data_participant_characteristics': 'Key participant characteristics',
    'data_n_participants': 'Number of participants (total in study)',
    'data_n_events': 'Number of outcomes/events (total events in study)',
    'data_epv': 'Number of outcomes/events wrt the number of candidate predictors (EPV)',
    'data_n_missing': 'Number of participants with any missing value',
    'data_missing_handling': 'Handling of missing data (e.g., complete-case analysis)',
    'data_splitting_strategy': 'Data splitting strategy',
    'data_splitting_ratios': 'Data splitting ratios',
    
    # Outcome
    'outcome_predicted': 'Outcome(s) predicted',
    'outcome_definition': 'Definition of each possible outcome',
    'outcome_assessment': 'How was outcome assessed (e.g., clinical diagnosis, microbiology, ICD codes)',
    'outcome_prevalence': 'Prevalence of each outcome',
    'outcome_def_same': 'Was the outcome definition the same for all participants? (yes, no)',
    'outcome_type': 'Type of outcome (e.g., single, combined, survival, binary, multi-class)',
    'outcome_blinded': 'Was the outcome assessed without knowledge of the candidate predictors? (yes, no)',
    'outcome_timing': 'Timing of outcome measurement/occurence (e.g., follow-up duration, time to event)',
    
    # Predictors
    'predictors_number': 'Number of (candidate) predictors assessed',
    'predictors_type': 'Type of predictors',
    'predictors_timing': 'Timing of predictor measurement',
    'predictors_similar': 'Were predictors and measurements similar for all participants? (yes, no)',
    'predictors_blinded': 'Were predictors assessed blinded for outcome? (yes, no)',
    'predictors_continuous_handling': 'Handling of continuous predictors in the modelling',
    'predictors_candidate_selection': 'Method for selection of candidate predictors',
    'predictors_selection': 'Method for selection of predictors during multivariable modelling',
    'predictors_imbalance_handling': 'Handling of class imbalance (e.g., oversampling)',
    'predictors_in_outcome': 'Were candidate predictors part of the outcome? (yes, no)',
    'predictors_n_final': 'Number of predictors/features used in the final model',
    
    # Model
    'models_used': 'Model name',
    'model_type': 'Model type',
    'model_software': 'Software/programming language/libraries used',
    'model_hyperparameter_tuning': 'Hyperparameter tuning method',
    'model_code_available': 'Is model code available? (yes, no, on request)',
    'model_internal_validation': 'Internal validation strategy',
    'model_external_validation': 'External evaluation strategy',
    'model_best_main': 'Main or best model',
    'model_architecture_provided': 'Final model architecture/equation provided',
    'model_n_parameters': 'Number of parameters in the final model',
    'model_baseline_used': 'Was a baseline model/method used',
    'model_interpretability': 'Feature importance/interpretability method',
    'model_top_features': 'If above is not NI, top-N most important features',
    
    # Metrics
    'metrics_c_statistic': 'C-Statistic',
    'metrics_calibration': 'Calibration metrics',
    'metrics_clinical_utility': 'Clinical utility metrics',
    'metrics_auroc': 'AUROC graph / value',
    'metrics_auprc': 'AUPRC (PR) graph / value',
    'metrics_log_rank': 'Log-rank test',
    'metrics_risk_curves': 'Risk group curves',
    'metrics_accuracy': 'Accuracy',
    'metrics_balanced_accuracy': 'Balanced accuracy',
    'metrics_specificity': 'Specificity',
    'metrics_recall': 'Recall (sensitivity)',
    'metrics_precision': 'Precision (PPV)',
    'metrics_npv': 'Negative predicted value - NPV',
    'metrics_f1_score': 'F1-Score',
    'metrics_other': 'Other',
    
    # Study results
    'study_key_findings': 'Key findings of the study',
    'study_evidence_strength': 'Strength of evidence for drivers',
    'study_strengths': 'Study strengths',
    'study_limitations': 'Study limitations',
    'study_future_research': 'Future research directions',
    'study_clinical_implications': 'Clinical implications and utility',
    'study_implementation_barriers': 'Potential barriers to implementation',
    
    # Transplant-specific
    'transplant_type': 'Type of transplant',
    'transplant_post_time': 'Time post-transplant covered by prediction',
    'transplant_infections_predicted': 'Specific infections predicted',
    'transplant_immunosuppression': 'Immunosuppression regimen details',
    'transplant_biomarkers': 'Any specific biomarkers or transplant-related predictors used',
    'transplant_opportunistic_infections': 'Consideration of opportunistic infections (yes/no)',
    'transplant_risk_stratification': 'Patient stratification by risk (if applicable)',
}

# Normalize keys for lookup
NORMALIZED_LOOKUP = {re.sub(r'[^a-zA-Z0-9]', '', v.lower()): k for k, v in COLUMN_MAPPING_INVERSED.items()}
NORMALIZED_LOOKUP.update({re.sub(r'[^a-zA-Z0-9]', '', k.lower()): k for k in COLUMN_MAPPING_INVERSED.keys()})

# Groups for looping
CATEGORICAL_FEATURES = [
    'transplant_type', 'transplant_infections_predicted', 'transplant_post_time',
    'data_source', 'data_sites_region', 'data_type', 
    'model_software', 'model_external_validation', 'model_interpretability',
    'model_best_main', 'data_splitting_strategy'
]

QUALITATIVE_FEATURES = [
    'data_inclusion_criteria', 'data_exclusion_criteria', 
    'study_limitations', 'study_key_findings'
]

# Country to continent mapping for site distribution plot
COUNTRY_TO_CONTINENT = {
    'usa': 'North & South America', 'us': 'North & South America', 'united states': 'North & South America',
    'canada': 'North & South America', 'mexico': 'North & South America', 'portugal': 'Europe', 'uk': 'Europe',
    'united kingdom': 'Europe', 'england': 'Europe', 'poland': 'Europe', 'france': 'Europe', 'germany': 'Europe',
    'italy': 'Europe', 'spain': 'Europe', 'netherlands': 'Europe', 'switzerland': 'Europe', 'belgium': 'Europe',
    'austria': 'Europe', 'sweden': 'Europe', 'denmark': 'Europe', 'norway': 'Europe', 'china': 'Asia',
    'japan': 'Asia', 'korea': 'Asia', 'south korea': 'Asia', 'india': 'Asia', 'taiwan': 'Asia', 'thailand': 'Asia',
    'hong kong': 'Asia', 'australia': 'Oceania', 'new zealand': 'Oceania', 'brazil': 'North & South America',
    'argentina': 'South America', 'chile': 'South America', 'south africa': 'Africa', 'egypt': 'Africa',
    'europe': 'Europe', 'north america': 'North & South America', 'asia': 'Asia'
}

# How to group similar entries for certain features
ENTRY_MAPPING = {
    # --- Data Splitting Strategies ---
    "Random / Sample Split": ["Random Split", "Holdout", "Patient level split", "Independent Cohort", "Custom Split"],
    "Cross-Validation": ["Cross-Validation", "Cross-Validation (LOOCV)", "Cross-Validation (k-fold)"],
    "Temporal Split": ["Time-based Split", "Temporal"],

    # --- Machine Learning Models ---
    "Random Forest": ["Random Forest", "Survival Conditional Random Forest Learner"],
    "Decision Tree": ["Decision Tree", "Decision Tree (C&RT", "Binary Decision Tree)", "Survival Decision Tree (LTRCart)"],
    "Gradient Boosting": ["Gradient Boosting (XGB)", "Gradient Boosting (GBM)", "Gradient Boosting (GBR)"],
    "Ensemble / Hybrid": ["Ensemble (SuperLearner)", "Hybrid-DEA"],
    "Logistic Regression": ["Logistic Regression"],
    "SVM": ["SVM"],

    # --- Data Sources ---
    "Retrospective (Clinical/EHR)": [
        "Retrospective Clinical Data", "Retrospective EHR", "Retrospective Database",
        "Retrospective Cohort", "Case-Control Clinical Data",
    ],
    "Prospective": ["Prospective Clinical Data", "Prospective Case-Control Clinical Data"],
    "Bespoke Cohort": ["Bespoke Cohort", "Bespoke Cohort (Clinical Study)"],

    # --- External Validation ---
    "Independent Cohort": ["Independent Cohort", "Independent Cohort (Different Setting)"],
    "Temporal Validation": ["Temporal Set"],
    "Holdout Set": ["Holdout Set"],

    # --- Transplant Types & Infections (Optional Grouping) ---
    "Bacterial Infection": ["Infection (Bacterial)", "Infection (UTI)", "Sepsis"],
    "Viral Infection": ["Infection (Viral)"],
    "Fungal Infection": ["Infection (Fungal)"],
}


# How to process each feature for plotting
FEATURE_STRATEGIES = {
    # Take First (Single value per paper)
    'publication_year': 'take_first',
    'publication_journal': 'take_first',
    'data_source': 'take_first',
    'data_available': 'take_first',
    'data_recruitment_method': 'take_first',
    'data_sites_number': 'take_first',
    'outcome_type': 'take_first',
    'outcome_blinded': 'take_first',
    'outcome_def_same': 'take_first',
    'predictors_number': 'take_first',
    'predictors_similar': 'take_first',
    'predictors_blinded': 'take_first',
    'predictors_in_outcome': 'take_first',
    'data_n_participants': 'take_first',
    'data_n_events': 'take_first',
    'data_epv': 'take_first',
    'data_n_missing': 'take_first',
    'data_missing_handling': 'take_first',
    'predictors_imbalance_handling': 'take_first',
    'model_code_available': 'take_first',
    'model_best_main': 'take_first',  # "Main or best model"
    'model_n_parameters': 'take_first',
    'model_baseline_used': 'take_first',
    'transplant_opportunistic_infections': 'take_first',
    'transplant_risk_stratification': 'take_first',
    
    # Join (Combinations form unique categories)
    'data_type': 'join',
    'data_sites_region': 'join',
    'outcome_predicted': 'join',
    'outcome_definition': 'join',
    'outcome_timing': 'join',
    'predictors_timing': 'join',
    'predictors_continuous_handling': 'join',
    'data_splitting_strategy': 'join',
    'model_internal_validation': 'join',
    'model_external_validation': 'join',
    'transplant_type': 'join',
    'transplant_post_time': 'join',
    'transplant_infections_predicted': 'join',

    # Binarize (reported vs not reported)
    'metrics_c_statistic': 'binarize',
    'metrics_calibration': 'binarize',
    'metrics_clinical_utility': 'binarize',
    'metrics_auroc': 'binarize',
    'metrics_auprc': 'binarize',
    'metrics_log_rank': 'binarize',
    'metrics_risk_curves': 'binarize',
    'metrics_accuracy': 'binarize',
    'metrics_balanced_accuracy': 'binarize',
    'metrics_specificity': 'binarize',
    'metrics_recall': 'binarize',
    'metrics_precision': 'binarize',
    'metrics_npv': 'binarize',
    'metrics_f1_score': 'binarize',
    'metrics_other': 'binarize',

    # Aggregate (explode list: one paper can count for multiple bars)
    # Default for anything else, but explicitly listed here:
    'models_used': 'aggregate',
    'data_study_settings': 'aggregate',
    'data_inclusion_criteria': 'aggregate',
    'data_exclusion_criteria': 'aggregate',
    'outcome_assessment': 'aggregate',
    'predictors_type': 'aggregate',
    'model_software': 'aggregate',
    'model_hyperparameter_tuning': 'aggregate',
    'predictors_candidate_selection': 'aggregate',
    'predictors_selection': 'aggregate',
    'study_strengths': 'aggregate',
    'study_limitations': 'aggregate',
    'study_future_research': 'aggregate',
    'study_clinical_implications': 'aggregate',
    'study_implementation_barriers': 'aggregate',
    'transplant_biomarkers': 'aggregate',
    'transplant_immunosuppression': 'aggregate',
    'study_key_findings': 'aggregate',
    'study_evidence_strength': 'aggregate',
    'model_interpretability': 'aggregate',
    'model_top_features': 'aggregate',
}