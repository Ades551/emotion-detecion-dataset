import numpy as np

# mlps = [
#         "valence",
#         #"Arousal", 
#         "submissive_vs_dominant", 
#         "serious_vs_humorous", 
#         #"Vulnerable_vs._Emotionally_Detached", 
#         "confident_vs_hesitant", 
#         "warm_vs_cold",
#         "monotone_vs_expressive",
#         "high_pitched_vs_low_pitched",
#         "soft_vs_harsh",
#         #"Authenticity"
# ]

mlps = [
        "Valence",
        "Submissive_vs._Dominant", 
        "Serious_vs._Humorous",
        "Confident_vs._Hesitant", 
        "Warm_vs._Cold",
        "Monotone_vs._Expressive",
        "High-Pitched_vs._Low-Pitched",
        "Soft_vs._Harsh"
        ]

attributes = {
        "valence" : "BIPO",
        ###"Arousal" : "UNIPO",
        "submissive_vs_dominant" : "BIPO", 
        "serious_vs_humorous" : "BIPO", 
        ###"Vulnerable_vs._Emotionally_Detached" : "UNIPO", 
        "confident_vs_hesitant" : "BIPO", 
        "warm_vs_cold" : "UNIPO",
        "monotone_vs_expressive" : "UNIPO",
        "high_pitched_vs_low_pitched" : "BIPO",
        "soft_vs_harsh" : "BIPO",
        ###"Authenticity" : "UNIPO",
        "llm" : "UNIPO",
        "czech_roberta" : "UNIPO",
        "wav2vec" : "UNIPO"
}

no_emotion_attributes_avgs = {
    'valence': 0.08549851851851852, 
    'submissive_vs_dominant': 0.4711611111111111,
    'serious_vs_humorous': 0.8252203703703704,
    'confident_vs_hesitant': 1.2958003703703704,
    'warm_vs_cold': 0.09717407407407408,
    'monotone_vs_expressive': 1.7792751851851851,
    'high_pitched_vs_low_pitched': 1.9460248148148147,
    'soft_vs_harsh': 0.3234122222222222,
    'llm': 0.11851851851851852,
    'czech_roberta': 0.13628112055637218,
    'wav2vec': 0.40917405243273136
}

no_emotion_attributes_vars = {
    'valence': 0.15001958847736627, 
    'submissive_vs_dominant': 0.312089012345679, 
    'serious_vs_humorous': 0.17111272976680383, 
    'confident_vs_hesitant': 0.230061122085048, 
    'warm_vs_cold': 0.14904990397805212, 
    'monotone_vs_expressive': 0.410335720164609, 
    'high_pitched_vs_low_pitched': 0.1493731358024691, 
    'soft_vs_harsh': 0.2777701152263375, 
    'llm': 0.2089437585733882, 
    'czech_roberta': 0.22309724179969073, 
    'wav2vec': 0.47585426838473205
}

best_T = 0.894477672319289

def z_score(data, attributes, attr_averages, attr_variances):
    z_score = {}
    for attr, polarity in attributes.items():
        value = data[attr]
        if polarity == "BIPO":
            z_score[attr] = [abs(value-attr_averages[attr])/attr_variances[attr]]
        elif polarity == "UNIPO":
            z_score[attr] = [(value-attr_averages[attr])/attr_variances[attr]]
       
    return z_score

def calculate_avg_for_all_clips(attributes, data):
    X = np.array([list(data[attr]) for attr in attributes]).T
    return X.mean(axis=1)

def predict(merged_models_output):
    normalized_output = z_score(merged_models_output, attributes, no_emotion_attributes_avgs, no_emotion_attributes_vars)
    avg_attr_value = calculate_avg_for_all_clips(attributes, normalized_output)[0]
    if avg_attr_value > best_T:
        return True
    else:
        return False