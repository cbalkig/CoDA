import fnmatch
import json
import logging
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import matplotlib
from tensorboard.backend.event_processing import event_accumulator as tb_ea

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image
from tqdm import tqdm


# -------------------- Utilities --------------------

def _pkg_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_yaml_or_json(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # pip install pyyaml
        except Exception as e:
            raise RuntimeError("PyYAML is required to read YAML. pip install pyyaml") from e
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_path(x, base: Path) -> Path | None:
    if not x:
        return None
    p = Path(x)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


# -------------------- Main config --------------------

def load_main_config() -> dict:
    d = _pkg_dir()
    for name in (
            "config.yml", "config.yaml", "config.json",
            "exporter_config.yml", "exporter_config.yaml", "exporter_config.json",
    ):
        p = d / name
        if p.exists():
            cfg = _load_yaml_or_json(p)
            cfg["_config_dir"] = d
            logging.debug(f"[config] loaded: {p}")
            return cfg
    raise FileNotFoundError("No config found next to this script.")


def load_report_config() -> dict:
    d = _pkg_dir()
    for name in (
            "report.yml",
    ):
        p = d / name
        if p.exists():
            cfg = _load_yaml_or_json(p)
            logging.debug(f"[config] loaded: {p}")
            return cfg
    raise FileNotFoundError("No report config found next to this script.")


# -------------------- Colors --------------------

def load_colors_cfg(colors_cfg: dict | None, colors_file: Path | None):
    if colors_file:
        data = _load_yaml_or_json(colors_file)
    else:
        data = colors_cfg or {}

    default_cycle = data.get("default_cycle")
    rules_src = data.get("colors", data)

    def _norm_list(rules, key):
        vals = rules.get(key, [])
        out = []
        for e in vals if isinstance(vals, list) else []:
            if isinstance(e, dict) and isinstance(e.get("match"), str) and isinstance(e.get("color"), str):
                out.append({"match": e["match"], "color": e["color"]})
        return out

    return {
        "default_cycle": default_cycle if isinstance(default_cycle, list) else None,
        "rules": {
            "series": _norm_list(rules_src, "series"),
            "tags": _norm_list(rules_src, "tags"),
            "runs": _norm_list(rules_src, "runs"),
        }
    }


def ColorChooser(cfg):
    """
    Most-specific match wins (exact > more concrete pattern > shorter).
    Precedence: tag > series > run.
    """
    default_cycle = cfg.get("default_cycle")
    if default_cycle:
        try:
            from cycler import cycler
            plt.rcParams["axes.prop_cycle"] = cycler(color=default_cycle)
        except Exception:
            pass

    series_rules = cfg["rules"]["series"]
    tag_rules = cfg["rules"]["tags"]
    run_rules = cfg["rules"]["runs"]

    def _score(pat: str, key: str):
        if pat == key:
            return (2, len(pat))  # exact match
        solid = len(pat.replace("*", "").replace("?", ""))
        return (1, solid)

    def _best_match(rules, key):
        best = None
        best_score = (-1, -1)
        for r in rules:
            pat = r.get("match");
            col = r.get("color")
            if isinstance(pat, str) and isinstance(col, str) and fnmatch.fnmatchcase(key, pat):
                sc = _score(pat, key)
                if sc > best_score:
                    best = col;
                    best_score = sc
        return best

    def color_for(series_label: str, full_tag: str, run_name: str):
        return (
                _best_match(tag_rules, full_tag) or
                _best_match(series_rules, series_label) or
                _best_match(run_rules, run_name)
        )

    return color_for


# -------------------- Notes (captions) --------------------

def load_notes_cfg(notes_cfg: dict | list | None, notes_file: Path | None):
    data = _load_yaml_or_json(notes_file) if notes_file else (notes_cfg or [])
    if isinstance(data, dict):
        items = data.get("notes", data if isinstance(data, list) else [])
    else:
        items = data

    out = []
    for e in items if isinstance(items, list) else []:
        m = e.get("match");
        t = e.get("text");
        w = (e.get("where") or "bottom").lower()
        if isinstance(m, str) and isinstance(t, (str, int, float)):
            out.append({"match": m, "text": str(t), "where": "top" if w == "top" else "bottom"})
    return out


def note_for_page(notes_rules, page_key_str: str):
    for r in notes_rules:
        if fnmatch.fnmatchcase(page_key_str, r["match"]):
            return r
    return None


def write_caption(fig, note_rule):
    """
    Write a caption below the plot in figure coordinates.
    Reserves bottom space via tight_layout(rect=...).
    """
    if not note_rule:
        return
    txt = dedent(str(note_rule["text"])).strip()
    if not txt:
        return
    # Centered caption just below the axes area
    fig.text(
        0.5, 0.02, txt,  # 2% from bottom
        ha="center", va="bottom", fontsize=10, wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="0.6"),
    )


# -------------------- Axis limits (per page) --------------------

def load_axis_limits(cfg: dict):
    """
    y_axis_limits:
      - { match: "Metrics/Recall", ymin: 0.3, ymax: 0.9 }
    """
    rules = cfg.get("y_axis_limits", [])
    out = []
    for e in rules if isinstance(rules, list) else []:
        m = e.get("match");
        ymin = e.get("ymin");
        ymax = e.get("ymax")
        if isinstance(m, str):
            out.append({"match": m, "ymin": ymin, "ymax": ymax})
    return out


def axis_limits_for_page(rules, page_key: str):
    for r in rules:
        if fnmatch.fnmatchcase(page_key, r["match"]):
            return r["ymin"], r["ymax"]
    return None, None


# -------------------- TensorBoard helpers --------------------

def tb_size_guidance(max_scalars: int, max_images: int):
    SCALARS = getattr(tb_ea, "SCALARS", "scalars")
    IMAGES = getattr(tb_ea, "IMAGES", "images")
    COMPRESSED_HISTOGRAMS = getattr(tb_ea, "COMPRESSED_HISTOGRAMS", "compressedHistograms")
    HISTOGRAMS = getattr(tb_ea, "HISTOGRAMS", "histograms")
    AUDIO = getattr(tb_ea, "AUDIO", "audio")
    TENSORS = getattr(tb_ea, "TENSORS", "tensors")
    return {SCALARS: max_scalars, IMAGES: max_images, COMPRESSED_HISTOGRAMS: 0,
            HISTOGRAMS: 0, AUDIO: 0, TENSORS: 0}


def find_run_dirs(logdir: Path):
    logdir = logdir.expanduser().resolve()
    if not logdir.is_dir():
        raise FileNotFoundError(f"Logdir not found or not a directory: {logdir}")

    def has_events(d: Path) -> bool:
        return any(p.is_file() and p.name.startswith("events.out.tfevents") for p in d.glob("**/*"))

    run_dirs = []
    if has_events(logdir):
        run_dirs.append(logdir)
    for sub in sorted(logdir.iterdir()):
        if sub.is_dir() and has_events(sub):
            run_dirs.append(sub)

    seen, uniq = set(), []
    for d in run_dirs:
        if d not in seen:
            uniq.append(d);
            seen.add(d)
    return uniq


def load_event_accumulator(run_dir: Path, max_scalars=100000, max_images=5000):
    ea = tb_ea.EventAccumulator(str(run_dir), size_guidance=tb_size_guidance(max_scalars, max_images))
    ea.Reload()
    return ea


# -------------------- Grouping --------------------

def _split_levels_any(tag: str):
    if "/" in tag:
        return tag.split("/")
    return tag.split("_")


def page_key(tag: str, mode: str):
    parts = _split_levels_any(tag)
    if mode == "exact":
        return tag
    take = 2 if mode == "tb2" else 3
    if len(parts) >= take:
        return "/".join(parts[:take])
    return "/".join(parts)


def series_label(tag: str, mode: str):
    parts = _split_levels_any(tag)
    if mode == "tb2":
        head = parts[:2]
    elif mode == "tb3":
        head = parts[:3]
    else:
        return parts[-1] if parts else tag
    tail = parts[len(head):]
    if tail:
        return "/".join(tail)
    return parts[-1] if parts else tag


def order_pages(keys):
    return sorted(keys, key=lambda s: s.lower())


# -------------------- Page aliases --------------------

def load_alias_rules(cfg: dict):
    """
    page_aliases:
      - { match: "Metrics*Recall*", page: "Metrics/Recall" }
    """
    rules = cfg.get("page_aliases", [])
    out = []
    for e in rules if isinstance(rules, list) else []:
        m = e.get("match")
        p = e.get("page")
        if isinstance(m, str) and isinstance(p, str):
            out.append({"match": m, "page": p})
    return out


def alias_page_for(rules, raw_tag: str, computed_page: str) -> str | None:
    for r in rules:
        if fnmatch.fnmatchcase(raw_tag, r["match"]) or fnmatch.fnmatchcase(computed_page, r["match"]):
            return r["page"]
    return None


# -------------------- Label overrides --------------------

def load_label_overrides(cfg: dict):
    """
    label_overrides:
      runs:   - { match: "*source_train*",      as: "source_train" }
              - { match: "*source_validation*", as: "source_validation" }
      tags:   - { match: "Metrics/Recall",      as: "Recall" }
      series: - { match: "Acc",                 as: "accuracy" }
    Precedence: runs > tags > series > default(series_label(tag, mode))
    """
    section = cfg.get("label_overrides", {})

    def _norm(lst):
        out = []
        for e in lst if isinstance(lst, list) else []:
            m = e.get("match");
            a = e.get("as")
            if isinstance(m, str) and isinstance(a, str):
                out.append({"match": m, "as": a})
        return out

    return {
        "runs": _norm(section.get("runs", [])),
        "tags": _norm(section.get("tags", [])),
        "series": _norm(section.get("series", [])),
    }


def _apply_overrides(base_label: str, full_tag: str, run_name: str, ov):
    def _hit(lst, key):
        for r in lst:
            if fnmatch.fnmatchcase(key, r["match"]):
                return r["as"]
        logging.warning(f'No match found for - {key}')
        return None

    return _hit(ov["runs"], run_name) or _hit(ov["tags"], full_tag) or _hit(ov["series"], base_label) or base_label


# -------------------- Plotting --------------------

import matplotlib.ticker as ticker
import matplotlib.patheffects as pe


def plot_page(pdf: PdfPages, items, page: str, dpi: int, color_for, note_rule, mode: str, overrides, axis_limits_rules):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    plotted = False
    used_labels = set()
    all_steps, all_values = [], []

    # 1) Collect series in memory first (so we can detect duplicates)
    series = []
    for it in items:
        run_name, ea, tag = it["run"], it["ea"], it["tag"]
        try:
            events = ea.Scalars(tag)
        except KeyError:
            continue
        if not events:
            continue

        # dedupe by step
        by_step = {}
        for ev in events:
            by_step[ev.step] = ev.value
        if not by_step:
            continue

        steps = sorted(by_step)
        values = [by_step[s] for s in steps]
        all_steps.extend(steps)
        all_values.extend(values)

        base = series_label(tag, mode)
        final_label = _apply_overrides(base, tag, run_name, overrides)
        label = final_label if final_label not in used_labels else f"{final_label} • {run_name}"
        used_labels.add(label)

        color = color_for(final_label, tag, run_name)
        series.append({
            "run": run_name,
            "tag": tag,
            "label": label,
            "steps": steps,
            "values": values,
            "color": color
        })

    if not series:
        plt.close(fig)
        return

    # 2) Detect identical time series (same x & y)
    #    Use rounding to avoid floating noise
    def sig(steps, values, prec=12):
        return (tuple(steps), tuple(round(v, prec) for v in values))

    buckets = {}
    for s in series:
        buckets.setdefault(sig(s["steps"], s["values"]), []).append(s)

    # 3) Compute a small vertical offset for duplicates so they’re all visible
    vmin = min(all_values)
    vmax = max(all_values)
    vspan = max(vmax - vmin, 1e-12)
    eps = 0.002 * vspan  # ~0.2% of range

    linestyles = ["-", "--", ":", "-."]
    markers = ["", "o", "s", "D", "^"]

    max_markers = 50  # show at most this many markers on a line

    for key, group in buckets.items():
        k = len(group)
        for i, s in enumerate(group):
            steps = s["steps"]
            values = s["values"]

            # vertical offset for duplicates
            offset = (i - (k - 1) / 2.0) * eps if k > 1 else 0.0
            y = [v + offset for v in values]

            ls = linestyles[i % len(linestyles)] if k > 1 else "-"
            mk = markers[i % len(markers)] if k > 1 else ""

            # ✅ sparse markers
            if mk:
                step_interval = max(1, len(steps) // max_markers)
                marker_indices = list(range(0, len(steps), step_interval))
            else:
                marker_indices = None

            col = s["color"]

            ax.plot(
                steps, y,
                label=s["label"],
                linestyle=ls,
                linewidth=1.8,
                alpha=0.95,
                color=col if col else None,
                path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()],
                marker=mk if mk else None,
                markevery=marker_indices  # <-- only draw markers at these positions
            )
            plotted = True

    ax.set_title(page)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Epoch ticks every 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

    if all_steps:
        ax.set_xlim(min(all_steps), max(all_steps))

    ymin, ymax = axis_limits_for_page(axis_limits_rules, page)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin if ymin is not None else min(all_values),
                    ymax if ymax is not None else max(all_values))

    # Legend outside at the bottom (above caption)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=False
    )

    # Reserve space for legend + optional caption
    bottom_margin = 0.28 if note_rule else 0.18
    fig.tight_layout(rect=[0.04, bottom_margin, 0.98, 0.96])

    if note_rule:
        write_caption(fig, note_rule)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pil_image_page(pdf: PdfPages, image: Image.Image, title: str, dpi: int, note_rule):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.imshow(image);
    ax.axis("off");
    ax.set_title(title)

    if note_rule:
        fig.tight_layout(rect=[0.04, 0.16, 0.98, 0.96])
        write_caption(fig, note_rule)
    else:
        fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.96])

    pdf.savefig(fig)
    plt.close(fig)


# -------------------- Export (merged across runs) --------------------

def export_scalars_merged(pdf: PdfPages, all_runs, grouping: str, dpi: int, color_for, notes_rules, alias_rules,
                          overrides, axis_limits_rules):
    pages = defaultdict(list)

    for run_name, ea in all_runs:
        try:
            scalar_tags = list(ea.Tags().get("scalars", []))
        except Exception:
            scalar_tags = []
        for t in scalar_tags:
            computed = page_key(t, grouping)
            aliased = alias_page_for(alias_rules, t, computed) or computed
            if aliased == computed:
                logging.warning(f'No alias for ---> {computed}')
            pages[aliased].append({"run": run_name, "ea": ea, "tag": t})

    for page in order_pages(pages.keys()):
        note_rule = note_for_page(notes_rules, page)
        try:
            plot_page(pdf, pages[page], page, dpi=dpi, color_for=color_for,
                      note_rule=note_rule, mode=grouping, overrides=overrides,
                      axis_limits_rules=axis_limits_rules)
        except Exception as e:
            tqdm.write(f"[WARN] Could not plot page '{page}': {e}")

        # Heuristic warning for likely pair-missing pages
        labels = []
        for it in pages[page]:
            base = series_label(it["tag"], grouping)
            final_lbl = _apply_overrides(base, it["tag"], it["run"], overrides)
            labels.append(final_lbl.replace("/", "_"))
        has_train = any(re.search(r"train\b", lbl) for lbl in labels)
        has_valid = any(re.search(r"valid", lbl) for lbl in labels)
        if has_train ^ has_valid:
            miss = "source_validation" if has_train else "source_train"
            tqdm.write(f"[WARN] Page '{page}' might be missing '{miss}'")


def export_images(pdf: PdfPages, all_runs, dpi: int, notes_rules, max_images_per_tag: int):
    for run_name, ea in all_runs:
        try:
            image_tags = list(ea.Images().keys())
        except Exception:
            image_tags = []
        for tag in sorted(image_tags, key=str.lower):
            try:
                events = ea.Images(tag)
            except KeyError:
                continue
            if not events:
                continue
            take = events[:max_images_per_tag] if max_images_per_tag > 0 else events
            for ev in take:
                try:
                    img = Image.open(BytesIO(ev.encoded_image_string)).convert("RGB")
                except Exception:
                    continue
                max_w, max_h = 1920, 1080
                w, h = img.size
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    img = img.resize((int(w * scale), int(h * scale)))
                page = tag
                note_rule = note_for_page(notes_rules, page)
                add_pil_image_page(pdf, img, f"{run_name} — {tag} (step {ev.step})", dpi, note_rule)


# -------------------- Output path --------------------

def pdf_path_in_logdir(logdir: Path) -> Path:
    logdir = logdir.expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir / "tensorboard_export.pdf"


# -------------------- Main --------------------

def export(log_dir: Path):
    cfg = load_main_config()
    base = cfg.get("_config_dir", _pkg_dir())

    grouping = (str(cfg.get("grouping") or "tb2")).lower()
    dpi = int(cfg.get("dpi") or 120)
    max_images = int(cfg.get("max_images_per_tag") or 10)

    colors_file = _resolve_path(cfg.get("colors_file"), base)
    colors_cfg_inline = cfg.get("colors")
    colors_cfg = load_colors_cfg(colors_cfg_inline, colors_file)
    color_for = ColorChooser(colors_cfg)

    notes_file = _resolve_path(cfg.get("notes_file"), base)
    notes_cfg_inline = cfg.get("notes")
    notes_rules = load_notes_cfg(notes_cfg_inline, notes_file)

    alias_rules = load_alias_rules(cfg)
    label_overrides = load_label_overrides(cfg)
    axis_limits_rules = load_axis_limits(cfg)

    logging.debug(f"[config] logdir={log_dir}")
    logging.debug(f"[config] grouping={grouping}, dpi={dpi}, max_images_per_tag={max_images}")
    logging.debug(
        f"[config] colors: series={len(colors_cfg['rules']['series'])}, tags={len(colors_cfg['rules']['tags'])}, runs={len(colors_cfg['rules']['runs'])}")
    logging.debug(f"[config] notes: {len(notes_rules)}")
    logging.debug(f"[config] page_aliases: {len(alias_rules)}")
    logging.debug(
        f"[config] label_overrides: runs={len(label_overrides['runs'])}, tags={len(label_overrides['tags'])}, series={len(label_overrides['series'])}")

    run_dirs = find_run_dirs(log_dir)
    if not run_dirs:
        logging.debug(f"No TensorBoard runs found under: {log_dir}")
        return

    all_runs = []
    for run_dir in tqdm(run_dirs, desc="Loading runs", unit="run", leave=False):
        run_name = (run_dir.relative_to(log_dir).as_posix() if run_dir != log_dir else run_dir.name)
        try:
            ea = load_event_accumulator(run_dir)
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load {run_dir}: {e}")
            continue
        all_runs.append((run_name, ea))

    out_path = pdf_path_in_logdir(log_dir)
    with PdfPages(str(out_path)) as pdf:
        export_scalars_merged(pdf, all_runs, grouping=grouping, dpi=dpi,
                              color_for=color_for, notes_rules=notes_rules,
                              alias_rules=alias_rules, overrides=label_overrides,
                              axis_limits_rules=axis_limits_rules)
        export_images(pdf, all_runs, dpi=dpi, notes_rules=notes_rules, max_images_per_tag=max_images)
        try:
            info = pdf.infodict()
            info["Title"] = "TensorBoard Export"
            info["Author"] = "report.py"
        except Exception:
            pass

    logging.debug(f"✅ Exported to: {out_path}")


if __name__ == "__main__":
    cfg = load_report_config()

    log_dir = _resolve_path(cfg.get("logdir"), None)
    if not log_dir:
        raise ValueError("`log_dir` must be set in the config file.")

    export(Path(log_dir))
