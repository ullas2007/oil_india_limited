def get_rule_score(text):
    text = str(text).lower()
    
    categories = {
        "Gas-test failure": [
            "gas test not done", "gas test was not done", "gas test missing", 
            "gas test blank", "gas test page was blank", "gas test sheet was blank", 
            "gas test expired", "gas test sheet was expired", "gas test skipped", 
            "no gas test attached", "fresh gas test not done", "fresh lel test not done", 
            "lel not checked", "lel values were not recorded", "detector battery low", 
            "detector showed error", "gas detector error"
        ],
        "Hot work": [
            "welding", "grinding", "cutting", "gas cutting", 
            "tack welding", "grinder", "spark", "hot work"
        ],
        "Hydrocarbon exposure": [
            "hydrocarbon", "hc smell", "hc odour", "vapour", 
            "lpg", "condensate", "flammable", "crude", "oil leak", "spill"
        ],
        "Energy-isolation failure": [
            "loto missing", "loto not verified", "lockout tagout not verified", 
            "isolation not verified", "line not depressurized", "live electrical", 
            "energised", "energized"
        ]
    }

    score = 0
    evidence = []

    for category, terms in categories.items():
        matched_terms = [term for term in terms if term in text]
        if matched_terms:
            score += 4
            evidence.append(f"{category}: {', '.join(matched_terms)}")

    return score, evidence


def map_life_saving_rules(text):
    text = str(text).lower()
    rules = []

    if any(term in text for term in ["welding", "grinding", "cutting", "gas cutting", "tack welding", "grinder", "spark"]):
        rules.append("Hot Work")
    if any(term in text for term in ["loto", "lockout tagout", "isolation not verified", "line not depressurized", "live electrical", "energised", "energized"]):
        rules.append("Energy Isolation")
    if any(term in text for term in ["confined space", "vessel entry", "tank entry", "pit entry", "h2s", "hydrogen sulfide", "low oxygen", "oxygen deficiency"]):
        rules.append("Confined Space")
    if any(term in text for term in ["suspended load", "crane", "sling", "slings", "dropped object", "struck by", "forklift", "vehicle reversing"]):
        rules.append("Line of Fire")
    if any(term in text for term in ["scaffold", "ladder", "harness", "open edge", "work at height", "fall arrest"]):
        rules.append("Work at Height")

    return rules if rules else ["Not applicable"]