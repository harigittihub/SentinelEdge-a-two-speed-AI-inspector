# Frozen design (committed before running the test split)

- Vision: PatchCore-lite, WideResNet50 layer2+3, 10% random patch bank, threshold = max score on 10% held-out good train images
- Jev: typesafe-ai/jev via Vercel AI Gateway, questions in decision/questions.py (unchanged)
- Escalation: Jev action == human_review -> Claude (anthropic/claude-sonnet-5.5), 3 images (reference, test, heatmap overlay), no crop, prompt as in decision/claude_client.py
- Policy: Claude auto-resolves a category only if, on dev, it had >= 10 escalated images and <= 2 errors. Result: metal_nut only. Other categories stay human_review.
- Primary comparison on test: vision rule vs Jev alone vs Jev + Claude (frozen policy). Metrics: false alarms, missed defects, accuracy.
- Tried and not adopted: zoomed crop (screw errors 15 -> 12, below the pre-set bar of ~5).