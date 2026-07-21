"""在低清墨水密度图上寻找同图异表之间的宽低密度区域。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..common.models import Box


@dataclass(frozen=True)
class DensityBand:
    axis: str
    start: int
    end: int
    mean_density: float
    source: str = "raw"

    @property
    def center(self) -> int:
        return round((self.start + self.end - 1) / 2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _fill_short_interruptions(low: np.ndarray, maximum_length: int) -> np.ndarray:
    """允许表标题等少量墨水穿过大空白区，不让它把空白带截成两段。"""

    result = low.copy()
    for start, end in _runs(~low):
        if (
            end - start <= maximum_length
            and start > 0
            and end < len(low)
            and low[start - 1]
            and low[end]
        ):
            result[start:end] = True
    return result


def _profile_bands(
    profile: np.ndarray,
    axis: str,
    *,
    maximum_density: float,
    minimum_band_ratio: float,
    maximum_interrupt_ratio: float,
    edge_margin_ratio: float,
    content_density: float,
    maximum_band_mean: float,
    support_ratio: float,
    narrow_band_maximum_mean: float,
    narrow_band_content_density: float,
) -> list[DensityBand]:
    length = len(profile)
    minimum_band = max(3, round(length * minimum_band_ratio))
    # 低清图中标题造成的中断通常只有 1～2 像素。超过 2 像素后更可能已经
    # 进入正文或相邻单元格，因此即使图片更长也不继续放宽。
    maximum_interrupt = max(1, min(2, round(length * maximum_interrupt_ratio)))
    edge_margin = max(1, round(length * edge_margin_ratio))
    raw_low = profile <= maximum_density
    filled_low = _fill_short_interruptions(raw_low, maximum_interrupt)
    # 填补标题中断有可能把真正分隔带一路串到页面尾部。原始低谷和填补后的
    # 宽低谷都参与候选，后续再按平均密度、边缘和内容支撑统一筛选。
    range_sources: dict[tuple[int, int], set[str]] = {}
    for item in _runs(raw_low):
        range_sources.setdefault(item, set()).add("raw")
    for item in _runs(filled_low):
        range_sources.setdefault(item, set()).add("filled")
    bands: list[DensityBand] = []
    for (start, end), sources in sorted(range_sources.items()):
        if start < edge_margin or end > length - edge_margin:
            continue
        # 上一步允许少量高密度位置（例如跨过空白区的表标题）把空白带短暂
        # 打断。但“打断很短”不代表整条带仍然足够空，因此合并之后必须再
        # 看一次平均密度。真正的表间空白可以容纳少量标题，但整条带的平均
        # 墨水仍应很低；否则正文行、正文列会被误判成分表带。
        band_mean = float(profile[start:end].mean())
        if band_mean > maximum_band_mean:
            continue
        support = max(3, round(length * support_ratio))
        before = profile[max(0, start - support) : start]
        after = profile[end : min(length, end + support)]
        if before.size == 0 or after.size == 0:
            continue
        if (
            float(before.mean()) < content_density
            or float(after.mean()) < content_density
        ):
            continue
        band_width = end - start
        razor_clear = False
        if band_width < minimum_band:
            # 5% 密度图会把真实表间空白压到 2 像素。不能无条件把最小宽度
            # 降到 2，否则表内行距也会参与分表；只有“本身几乎全白，且上下
            # 都有很强内容”的孤立窄带才放行。
            exceptionally_clear = (
                band_width >= 2
                and band_mean <= narrow_band_maximum_mean
                and float(before.mean()) >= narrow_band_content_density
                and float(after.mean()) >= narrow_band_content_density
            )
            # 超宽图固定缩到 5% 后，原图几十像素的真实表间空白可能只剩
            # 1 个像素。此时不能再看绝对宽度，而要看它是否是一个非常深、
            # 且左右（横向分表时即上下）立刻回到内容区的“刀口低谷”。
            # 该条件比普通窄带严格得多，避免把表内普通行距拿来分表。
            immediate_before = float(profile[start - 1]) if start > 0 else 0.0
            immediate_after = float(profile[end]) if end < length else 0.0
            razor_clear = (
                band_width == 1
                and band_mean <= min(narrow_band_maximum_mean, 0.0025)
                and immediate_before >= content_density
                and immediate_after >= content_density
                and float(before.mean()) >= content_density
                and float(after.mean()) >= content_density
            )
            if not exceptionally_clear and not razor_clear:
                continue
        bands.append(
            DensityBand(
                axis=axis,
                start=start,
                end=end,
                mean_density=band_mean,
                source=(
                    "+".join([*sorted(sources), "razor"])
                    if razor_clear
                    else "+".join(sorted(sources))
                ),
            )
        )
    return bands


def _cluster_nearby_bands(
    bands: list[DensityBand], maximum_span: int
) -> list[list[DensityBand]]:
    """把同一处宽空白附近的数条窄低谷归为一个候选区。

    分表带附近偶尔还会有标题下方的短空行。这里以每组第一条低谷为锚点，
    避免使用链式合并把一整排单元格空隙串成一个超宽候选区。
    """

    if not bands:
        return []
    groups: list[list[DensityBand]] = []
    current = [bands[0]]
    anchor = bands[0].center
    for band in bands[1:]:
        if band.center - anchor <= maximum_span:
            current.append(band)
        else:
            groups.append(current)
            current = [band]
            anchor = band.center
    groups.append(current)
    return groups


def _select_axis_bands(
    bands: list[DensityBand],
    *,
    coarse_max_side: int,
    cluster_ratio: float,
    minimum_region_ratio: float,
) -> list[DensityBand]:
    """删除明显属于表内密集行列空隙的候选，并保留每个分隔区的主低谷。"""

    groups = _cluster_nearby_bands(
        bands, max(4, round(coarse_max_side * cluster_ratio))
    )
    groups = [
        group
        for group in groups
        if not (len(group) >= 3 and all("filled" not in band.source for band in group))
    ]
    selected: list[DensityBand] = []
    for group in groups:
        # 横向表间区域如果被下一张表的标题切成上下两段，切口必须落在
        # 标题上方，让标题跟随下面的表。filled候选仍用于确认这是同一个
        # 宽分隔区，但不能再用它跨过标题后从整段中央落刀。
        raw_horizontal = [
            band for band in group if band.axis == "horizontal" and band.source == "raw"
        ]
        if raw_horizontal:
            selected.append(min(raw_horizontal, key=lambda band: band.start))
        else:
            selected.append(
                max(group, key=lambda band: (band.end - band.start, -band.mean_density))
            )
    # 1像素“刀口”只用于185a这类少量孤立低谷。若同一方向连续出现
    # 3条以上，通常是长表每隔若干行重复一次的内部留白（d8b即如此）；
    # 此时撤销所有刀口候选，但仍保留原本达到常规宽度的分表带。
    razor_count = sum("razor" in band.source for band in selected)
    if razor_count >= 3:
        selected = [band for band in selected if "razor" not in band.source]

    if len(selected) >= 2:
        spacing = float(np.median(np.diff([band.center for band in selected])))
        # 如果候选低谷像单元格一样密集重复，它们是内部行列而不是同图异表。
        if spacing < coarse_max_side * minimum_region_ratio:
            return []

    return selected


def find_density_bands(
    density: np.ndarray,
    *,
    maximum_density: float = 0.02,
    minimum_band_ratio: float = 0.007,
    maximum_interrupt_ratio: float = 0.008,
    edge_margin_ratio: float = 0.025,
    content_density: float = 0.03,
    maximum_band_mean: float = 0.015,
    support_ratio: float = 0.03,
    narrow_band_maximum_mean: float = 0.005,
    narrow_band_content_density: float = 0.20,
    cluster_ratio: float = 0.08,
    minimum_region_ratio: float = 0.12,
) -> tuple[list[DensityBand], list[DensityBand]]:
    """返回单一主方向上的分表带，不把密集单元格空隙当成同图异表。"""

    row_profile = density.mean(axis=1)
    rows = _profile_bands(
        row_profile,
        "horizontal",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
        maximum_band_mean=maximum_band_mean,
        support_ratio=support_ratio,
        narrow_band_maximum_mean=narrow_band_maximum_mean,
        narrow_band_content_density=narrow_band_content_density,
    )
    columns = _profile_bands(
        density.mean(axis=0),
        "vertical",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
        maximum_band_mean=maximum_band_mean,
        support_ratio=support_ratio,
        narrow_band_maximum_mean=narrow_band_maximum_mean,
        narrow_band_content_density=narrow_band_content_density,
    )
    coarse_max_side = max(density.shape)
    rows = _select_axis_bands(
        rows,
        coarse_max_side=coarse_max_side,
        cluster_ratio=cluster_ratio,
        minimum_region_ratio=minimum_region_ratio,
    )
    # filled候选说明一整段表间空白被少量标题墨迹打断。最终切口改放到
    # 第一段标题前的纯空白中，保证下表标题不会再被分给上一张表。
    minimum_band = max(3, round(len(row_profile) * minimum_band_ratio))
    adjusted_rows: list[DensityBand] = []
    for band in rows:
        interruptions = _runs(row_profile[band.start : band.end] > maximum_density)
        if "filled" in band.source and interruptions:
            end = band.start + interruptions[0][0]
            if end - band.start >= minimum_band:
                adjusted_rows.append(
                    DensityBand(
                        "horizontal",
                        band.start,
                        end,
                        float(row_profile[band.start : end].mean()),
                        "raw-before-title",
                    )
                )
                continue
        adjusted_rows.append(band)
    rows = adjusted_rows
    columns = _select_axis_bands(
        columns,
        coarse_max_side=coarse_max_side,
        cluster_ratio=cluster_ratio,
        minimum_region_ratio=minimum_region_ratio,
    )
    row_score = sum(
        (band.end - band.start) * (maximum_band_mean - band.mean_density)
        for band in rows
    )
    column_score = sum(
        (band.end - band.start) * (maximum_band_mean - band.mean_density)
        for band in columns
    )
    if row_score >= column_score:
        return rows, []
    return [], columns


def _merge_horizontal_gap_fragments(raw_gaps, row_density):
    """把被窄标题或表头打断的上下两段白缝重新合并。"""
    if not raw_gaps:
        return []
    max_bridge=max(2,round(len(row_density)*0.02))
    typical=float(np.median([b-a for a,b in raw_gaps]))
    side_min=max(2,round(typical*0.75))
    merged=[]; index=0
    while index<len(raw_gaps):
        start,end=raw_gaps[index]
        while index+1<len(raw_gaps):
            next_start,next_end=raw_gaps[index+1]
            bridge=next_start-end
            if len(raw_gaps)<=12 and typical>=4.0 and end-start>=side_min and next_end-next_start>=side_min and 0<bridge<=max_bridge:
                end=next_end; index+=1
            else:
                break
        merged.append((start,end)); index+=1
    return merged

def _normalize_gap_ranges(ranges):
    normalized=[]
    for start,end in sorted(ranges):
        if end<=start: continue
        if normalized and start<=normalized[-1][1]:
            normalized[-1]=(normalized[-1][0],max(normalized[-1][1],end))
        else:
            normalized.append((start,end))
    return normalized

def _horizontal_segment_owners(flags):
    bodies=[i for i,value in enumerate(flags) if value]
    if not bodies: return [0 for _ in flags]
    owners=[]
    for index,is_body in enumerate(flags):
        if is_body: owners.append(index); continue
        next_body=next((i for i in bodies if i>index),None)
        previous=next((i for i in reversed(bodies) if i<index),None)
        owners.append(next_body if next_body is not None else previous)
    return owners

def _safe_horizontal_cut(gap,raw_gaps):
    contained=[raw for raw in raw_gaps if gap[0]<=raw[0] and raw[1]<=gap[1]]
    selected=contained[0] if len(contained)>=2 else gap
    return round((selected[0]+selected[1]-1)/2)

def horizontal_table_split_boxes(
    preview,
    *,
    gray_threshold=225,
    horizontal_smear_ratio=0.01,
    blank_row_ratio=0.01,
):
    """只按上下方向分表；返回20%分析图坐标中的分表框。"""
    gray=np.asarray(preview.convert("L"))
    ink=(gray<gray_threshold).astype(np.uint8)
    smear_width=max(3,round(preview.width*horizontal_smear_ratio))
    smeared=cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_RECT,(smear_width,1)))
    row_density=smeared.mean(axis=1)
    hard_content=row_density>blank_row_ratio
    content_runs=_runs(hard_content)
    if not content_runs:
        return [Box(0,0,preview.width,preview.height)],[],{"raw_gaps":[],"final_gaps":[]}
    content_start,content_end=content_runs[0][0],content_runs[-1][1]
    raw_gaps=[(a,b) for a,b in _runs(~hard_content) if content_start<a and b<content_end]
    merged=_merge_horizontal_gap_fragments(raw_gaps,row_density)
    soft=[(a,b) for a,b in _runs((row_density<=0.05)&hard_content) if b-a<=8]
    enabled_soft=[gap for gap in soft if len(raw_gaps)>12 and content_start<gap[0] and gap[1]<content_end]
    all_gaps=_normalize_gap_ranges(merged+enabled_soft)
    base=merged or enabled_soft
    widths=[b-a for a,b in base]
    typical=float(np.median(widths)) if widths else 0.0
    relaxed=len(raw_gaps)<=12 and typical>=5.0
    minimum=max(3,round(typical*(1.0 if relaxed else 2.0)))
    candidates=[gap for gap in all_gaps if gap[1]-gap[0]>=minimum]
    edges=[content_start,*[v for gap in candidates for v in gap],content_end]
    segment_ranges=[(edges[i],edges[i+1]) for i in range(0,len(edges)-1,2)]
    minimum_height=max(8,round(preview.height*0.02))
    minimum_active=max(6,minimum_height//3)
    segments=[]
    for start,end in segment_ranges:
        height=end-start; active=int(hard_content[start:end].sum())
        segments.append((start,end,height,active,height>=minimum_height and active>=minimum_active))
    owners=_horizontal_segment_owners([item[4] for item in segments])
    final=[candidates[i] for i in range(len(candidates)) if owners[i] is not None and owners[i+1] is not None and owners[i]!=owners[i+1]]
    for index,gap in enumerate(candidates):
        if gap in final: continue
        left,right=segments[index],segments[index+1]
        for small,body in ((left,right),(right,left)):
            if small[4] or not body[4]: continue
            small_density=float(row_density[small[0]:small[1]].mean())
            body_density=float(row_density[body[0]:body[1]].mean())
            enough_height=small[2]>=max(8,round(minimum_height*0.5))
            enough_ink=small[3]>=minimum_active
            table_like=small_density>=0.15 and small_density>=body_density*0.55
            if enough_height and enough_ink and table_like:
                final.append(gap); break
    final=sorted(set(final))
    cuts=[_safe_horizontal_cut(gap,raw_gaps) for gap in final]
    boundaries=[content_start,*cuts,content_end]
    boxes=[Box(0,y1,preview.width,y2) for y1,y2 in zip(boundaries,boundaries[1:]) if y2>y1]
    bands=[DensityBand("horizontal",a,b,float(row_density[a:b].mean()),"horizontal-v2") for a,b in final]
    return boxes,bands,{"raw_gaps":raw_gaps,"merged_gaps":merged,"soft_gaps":soft,"candidate_gaps":candidates,"final_gaps":final,"cut_positions":cuts,"row_density":row_density}

def boxes_from_bands(
    width: int,
    height: int,
    horizontal: list[DensityBand],
    vertical: list[DensityBand],
    density: np.ndarray,
    minimum_content_density: float = 0.01,
) -> list[Box]:
    """按候选带中心形成网格，并删除几乎没有内容的格子。"""

    xs = [0, *sorted({band.center for band in vertical}), width]
    ys = [0, *sorted({band.center for band in horizontal}), height]
    boxes: list[Box] = []
    for y1, y2 in zip(ys, ys[1:]):
        for x1, x2 in zip(xs, xs[1:]):
            if x2 <= x1 or y2 <= y1:
                continue
            if float(density[y1:y2, x1:x2].mean()) < minimum_content_density:
                continue
            boxes.append(Box(x1, y1, x2, y2))
    return boxes
