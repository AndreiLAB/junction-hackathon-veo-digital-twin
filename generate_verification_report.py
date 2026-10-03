import json

with open('outputs/raw_detections.json') as f:
    raw = json.load(f)

with open('outputs/physical_tags.json') as f:
    phys = json.load(f)

lines = [
    '# Vision Detector Verification Report\n',
    '| Raw Tag ID | Class / Label | Type | Position (X, Y, Z) | Confidence | Physical Tag ID | Obs Count |',
    '|---|---|---|---|---|---|---|'
]

for r in raw:
    rx = r['position']['x']
    ry = r['position']['y']
    rz = r['position']['z']
    label = r['label']
    
    # find closest physical tag with same label
    best_pid = 'None'
    best_dist = 999.0
    best_obs = 1
    for p in phys:
        if p['label'] == label:
            dx = p['position']['x'] - rx
            dy = p['position']['y'] - ry
            dz = p['position']['z'] - rz
            dist = (dx**2 + dy**2 + dz**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_pid = p['tag_id']
                best_obs = p['observation_count']
                
    raw_id = r['tag_id'][:8]
    p_id = best_pid[:8] if best_pid != 'None' else 'None'
    lines.append(f"| `{raw_id}` | **{label}** | {r['type']} | `({rx:.2f}, {ry:.2f}, {rz:.2f})` | {r['confidence']:.2f} | `{p_id}` | {best_obs} |")

with open('outputs/verification_report.md', 'w') as f:
    f.write('\n'.join(lines))

print(f"Generated verification report with {len(raw)} entries in outputs/verification_report.md")
