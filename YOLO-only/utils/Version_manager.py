import argparse
import json
from pathlib import Path
from datetime import datetime
import sys


class VersionManager:
    """Version management command-line tool"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.version_file = self.models_dir / "version_history.json"

        if not self.version_file.exists():
            print(f"❌ No version history found in {output_dir}")
            sys.exit(1)

        with open(self.version_file, 'r') as f:
            self.history = json.load(f)

    def list_versions(self):
        """List all versions"""
        print("\n" + "=" * 80)
        print("📚 MODEL VERSION HISTORY")
        print("=" * 80)

        if not self.history["versions"]:
            print("No versions yet.")
            return

        current = self.history["current_version"]

        print(f"\n🎯 Current Version: v{current}\n")

        for v_info in self.history["versions"]:
            version = v_info["version"]
            timestamp = v_info["timestamp"]
            metrics = v_info["metrics"]
            dataset = v_info.get("dataset", {})

            marker = "⭐" if version == current else "  "
            print(f"{marker} Version {version}")
            print(f"   📅 Date: {timestamp}")

            if metrics:
                print(f"   📊 Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"      - {key}: {value:.4f}")
                    else:
                        print(f"      - {key}: {value}")

            if dataset:
                print(f"   📁 Dataset:")
                for key, value in dataset.items():
                    print(f"      - {key}: {value}")

            print()

        print("=" * 80)

    def compare_versions(self, v1: int, v2: int):
        """Compare the two versions"""
        versions = {v["version"]: v for v in self.history["versions"]}

        if v1 not in versions:
            print(f"❌ Version {v1} not found")
            return

        if v2 not in versions:
            print(f"❌ Version {v2} not found")
            return

        print("\n" + "=" * 80)
        print(f"📊 COMPARING v{v1} vs v{v2}")
        print("=" * 80)

        metrics1 = versions[v1]["metrics"]
        metrics2 = versions[v2]["metrics"]

        print(f"\n📅 v{v1} Date: {versions[v1]['timestamp']}")
        print(f"📅 v{v2} Date: {versions[v2]['timestamp']}")

        print(f"\n📈 Performance Changes:\n")

        for key in metrics1:
            if key in metrics2 and isinstance(metrics1[key], (int, float)):
                val1 = metrics1[key]
                val2 = metrics2[key]
                diff = val2 - val1
                percent = (diff / val1 * 100) if val1 != 0 else 0

                if diff > 0:
                    symbol = "📈 ↑"
                    color = "\033[92m"  # Green
                elif diff < 0:
                    symbol = "📉 ↓"
                    color = "\033[91m"  # Red
                else:
                    symbol = "➡️  ="
                    color = "\033[93m"  # Yellow

                reset = "\033[0m"

                print(f"   {symbol} {key}:")
                print(f"      v{v1}: {val1:.4f}")
                print(f"      v{v2}: {val2:.4f}")
                print(f"      {color}Change: {diff:+.4f} ({percent:+.2f}%){reset}")
                print()

        print("=" * 80)

    def show_best_version(self):
        """Display the best version"""
        if not self.history["versions"]:
            print("No versions yet.")
            return

        # Find the highest version of mAP@50
        best_version = max(
            self.history["versions"],
            key=lambda v: v["metrics"].get("mAP50", 0)
        )

        version = best_version["version"]
        mAP50 = best_version["metrics"].get("mAP50", 0)

        print("\n" + "=" * 80)
        print("🏆 BEST VERSION BY mAP@0.5")
        print("=" * 80)
        print(f"\n⭐ Version {version}")
        print(f"   mAP@0.5: {mAP50:.4f}")
        print(f"   Date: {best_version['timestamp']}")
        print(f"   Path: {best_version['model_path']}")
        print("\n" + "=" * 80)

    def export_version_info(self, output_file: str):
        """The export version information is Markdown"""
        with open(output_file, 'w') as f:
            f.write("# Model Version History\n\n")
            f.write(f"**Current Version:** v{self.history['current_version']}\n\n")
            f.write(f"**Total Versions:** {len(self.history['versions'])}\n\n")

            f.write("## Version Details\n\n")

            for v_info in self.history["versions"]:
                version = v_info["version"]
                timestamp = v_info["timestamp"]
                metrics = v_info["metrics"]

                f.write(f"### Version {version}\n\n")
                f.write(f"- **Date:** {timestamp}\n")
                f.write(f"- **Model Path:** `{v_info['model_path']}`\n\n")

                f.write("**Metrics:**\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"| {key} | {value:.4f} |\n")
                    else:
                        f.write(f"| {key} | {value} |\n")

                f.write("\n---\n\n")

        print(f"✅ Version info exported to {output_file}")

    def rollback_to_version(self, version: int):
        """Roll back to the specified version"""
        versions = {v["version"]: v for v in self.history["versions"]}

        if version not in versions:
            print(f"❌ Version {version} not found")
            return

        print(f"\n⚠️  Rolling back to version {version}...")

        # Update current_version
        self.history["current_version"] = version

        with open(self.version_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Update symbolic links
        latest_link = self.models_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(f"v{version}", target_is_directory=True)

        best_latest = self.models_dir / "best_latest.pt"
        if best_latest.exists():
            best_latest.unlink()
        best_latest.symlink_to(f"v{version}/weights/best.pt")

        print(f"✅ Rolled back to v{version}")
        print(f"🔗 Updated links:")
        print(f"   - models/latest -> v{version}")
        print(f"   - models/best_latest.pt -> v{version}/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Model Version Management Tool")
    parser.add_argument('--output_dir', type=str, default='incremental_output',
                        help="Output directory containing version history")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list command
    subparsers.add_parser('list', help='List all versions')

    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('v1', type=int, help='First version')
    compare_parser.add_argument('v2', type=int, help='Second version')

    # best command
    subparsers.add_parser('best', help='Show best version')

    # export command
    export_parser = subparsers.add_parser('export', help='Export version info to Markdown')
    export_parser.add_argument('--output', type=str, default='version_history.md',
                               help='Output file')

    # rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to a version')
    rollback_parser.add_argument('version', type=int, help='Version to rollback to')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    vm = VersionManager(args.output_dir)

    if args.command == 'list':
        vm.list_versions()
    elif args.command == 'compare':
        vm.compare_versions(args.v1, args.v2)
    elif args.command == 'best':
        vm.show_best_version()
    elif args.command == 'export':
        vm.export_version_info(args.output)
    elif args.command == 'rollback':
        vm.rollback_to_version(args.version)


if __name__ == "__main__":
    main()