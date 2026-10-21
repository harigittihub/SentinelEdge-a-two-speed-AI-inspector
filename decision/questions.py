QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Decide what to do with this inspected part, based on the anomaly measurements.",
        "criteria": {
            "pass": "no meaningful anomaly; the part is acceptable",
            "rework": "minor or localized defect that can be repaired or cleaned",
            "scrap": "large or serious defect; the part must be discarded",
            "human_review": "measurements are borderline or ambiguous; a person should look",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "Rate the severity of the defect on this part.",
        "criteria": [
            "none: no anomaly",
            "minor: small cosmetic anomaly",
            "major: large or functional defect",
            "critical: severe defect, line or safety risk",
        ],
    },
    "stopLine": {
        "type": "boolean",
        "instructions": "Should the production line be stopped?",
        "criteria": {
            "true": "severe or repeated-looking defect that risks more bad parts",
            "false": "isolated or minor issue; production can continue",
        },
    },
}