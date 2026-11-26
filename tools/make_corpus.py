"""Generate a training corpus of Minecraft and mod-loader style English lines.

Each line is built from a template with random mod names, versions, instance
names, and paths, so the trainer sees varied text. Output is one line per row.

Usage (from the repo folder):
    python tools/make_corpus.py --count 500 --out data/corpus.txt
"""

import argparse
import random
from pathlib import Path

MODS = [
    "glassseam-fix", "jei", "create", "journeymap", "applied-energistics-2",
    "mekanism", "sodium", "iris", "oculus", "waystones", "carpet", "fabric-api",
    "cloth-config", "modmenu", "terralith", "biomesoplenty", "farmersdelight",
    "alexsmobs", "quark", "ironchests", "supplementaries", "xaeros-minimap",
]
LIBRARIES = [
    "net.minecraftforge.fmlcore", "net.minecraftforge.javafmlmod",
    "org.spongepowered.mixin", "com.google.guava", "org.apache.logging.log4j",
    "net.fabricmc.loader", "com.electronwizard.config", "org.lwjgl.glfw",
]
INSTANCES = [
    "1.20.1 6June2026Wiped9Nov2026", "1.20.1 Forge", "1.19.2 Modded",
    "1.20.4 Fabric", "1.16.5 Create", "1.21 Survival", "Server Backup 2026",
]
DRIVES = ["D:", "C:", "E:"]
VERSIONS = ["1.20.1", "1.19.2", "1.16.5", "1.20.4", "1.21", "47.4.10", "0.14.22", "5.4.0"]
BUILDS = ["47.4.10", "47.2.0", "0.15.3", "3.1.8", "2.0.0-beta.4"]
PORTS = ["25565", "25566", "8080"]

TEMPLATES = [
    "The Mod File {drive}\\Games\\MultiMC\\instances\\{instance}\\.minecraft\\mods\\{mod}.jar has mods that were not found",
    "Failed to load mod {mod} {build}: missing dependency {lib}",
    "Missing dependency: {lib} version {version} required by {mod}",
    "Error loading class {lib}.{mod} in {mod}-{version}.jar",
    "Could not find {lib} for {mod}, skipping",
    "Mod {mod} requires Minecraft {version} but found {build}",
    "Loading mod list from {drive}\\Games\\MultiMC\\instances\\{instance}\\mods",
    "Skipping {mod}: incompatible with forge {build}",
    "Crash report saved to {drive}\\Games\\MultiMC\\crash-reports\\crash-2026-11-{day}.txt",
    "Server started on port {port} after {seconds} seconds",
    "Connection timed out while reading packet {packet}",
    "Texture pack reloaded with {count} resources",
    "Saved world settings to disk successfully",
    "Launching {instance} with {ram} MB of memory",
    "java.lang.NoClassDefFoundError: {lib}/{mod}",
    "[WARN] Config file {mod}-common.toml is out of date",
    "[INFO] Registered {count} recipes from {mod}",
    "Downloaded {mod}-{version}.jar in {seconds} seconds",
    "Set JVM argument -Xmx{ram}M for {instance}",
    "Retrying download of {mod}: HTTP status {status}",
    "{mod} {version} is already installed in {instance}",
    "Deleted {count} old log files from {drive}\\Games\\MultiMC\\logs",
    "Found {count} mods, {failed} failed to load",
    "Unable to open {drive}\\Games\\MultiMC\\libraries\\{lib_path}\\{build}\\{mod}-{build}.jar",
    "Mixin apply failed for {lib}: target class not found",
    "Chunk loading took {seconds} ms in dimension minecraft:overworld",
]


def make_line(rng, template):
    lib = rng.choice(LIBRARIES)
    return template.format(
        drive=rng.choice(DRIVES),
        instance=rng.choice(INSTANCES),
        mod=rng.choice(MODS),
        lib=lib,
        lib_path=lib.replace(".", "\\"),
        version=rng.choice(VERSIONS),
        build=rng.choice(BUILDS),
        port=rng.choice(PORTS),
        seconds=rng.randint(1, 240),
        day=f"{rng.randint(1, 30):02d}",
        packet=rng.randint(0x10, 0xFF),
        count=rng.randint(2, 900),
        ram=rng.choice([2048, 4096, 6144, 8192]),
        status=rng.choice([403, 404, 500, 503]),
        failed=rng.randint(0, 9),
    )


def generate(count, seed):
    rng = random.Random(seed)
    lines = []
    seen = set()
    attempts = 0
    while len(lines) < count and attempts < count * 20:
        attempts += 1
        line = make_line(rng, rng.choice(TEMPLATES))
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate a Minecraft-style training corpus.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="data/corpus.txt")
    args = parser.parse_args()

    lines = generate(args.count, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()
