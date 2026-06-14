from pathlib import Path

def generate_report(predictions, output_path="reports/inference_report.md"):
    """Generate human-readable clinical report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# INFERENCE REPORT\n\n")
        f.write("## Summary Statistics\n\n")

        from collections import Counter
        overall_counts = Counter([p['overall'] for p in predictions])
        f.write(f"| Class | Count | Percentage |\n")
        f.write(f"|-------|-------|------------|\n")
        for cls in ["CVR", "BRT", "PVR", "VBR", "TRF", "SMR"]:
            c = overall_counts.get(cls, 0)
            pct = 100 * c / len(predictions)
            f.write(f"| {cls} | {c} | {pct:.1f}% |\n")

        f.write("\n## Sample Predictions\n\n")
        for i, p in enumerate(predictions[:5]):
            f.write(f"### Trajectory #{i+1}\n\n")
            f.write(f"- **Predicted Class:** {p['overall']} ({p['overall_confidence']*100:.1f}% confidence)\n")
            f.write(f"- **Mechanism:** {p['mechanism']} ({p['mechanism_confidence']*100:.1f}% confidence)\n")
            f.write(f"- **Subsystem Health:**\n")
            for sub, score in p['subsystem_health'].items():
                status = "✓" if score > 0.5 else "⚠"
                f.write(f"  - {sub}: {score:.2f} {status}\n")
            f.write("\n")

    print(f"Report saved to {output_path}")
