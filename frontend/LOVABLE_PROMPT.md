# Lovable prompt: AutoTag360 frontend (paste everything below the line into Lovable)

---

Build **AutoTag360**, a web app that automatically tags the cabinets of a switchgear hall in a digital twin. It talks to an existing **REST backend** (spec below). **Do not invent data and do not add mock data to the working path:** everything shown comes from the API. If a value is `null`, show "—". Stack: React + TypeScript + Tailwind + shadcn/ui, React Query for data fetching. Responsive (laptop first, works on a tablet).

## 1. What the product does

A user picks **one of two methods** and gets, for every cabinet found, a **tag** (name), a **picture**, and the **manuals (PDFs)**, plus an **export** they can use to create the tags in the digital twin (a manual tagging sheet, and Matterport-compatible JSON).

- **Method A: E57 scan.** The scan is already processed on the server, so there is **no upload**. A button shows the cabinets the E57 pipeline found.
- **Method B: Image + coordinates.** The user selects a photo (or types the camera position and orientation). The app shows which cabinets are in view and where each asset (panel, relay, breaker window) must appear. **If no cabinet is in view, no tag is created** and the UI says so.

Assets and manuals: a UniGear ZS2 **panel** -> UniGear ZS2 manual; an ABB 615 **relay** -> REX615 manual; a **VD4 breaker** (seen through a window) -> VD4 manual. Cabinets H01-H05 are UniGear panels (shown as **assumed** until a detection confirms it).

## 2. Backend connection

- A **Settings** dialog (gear icon, top right) with **API base URL** (default from `VITE_API_BASE_URL`, saved in localStorage) and a connection indicator that calls `GET /health` (green = ok, red = unreachable, with the error text).
- All image/PDF links from the API are **relative paths** (`/assets/context/H05.jpg`, `/documents/2/file`, `/photos/img_067.jpg/image`): prefix them with the base URL.
- Show a friendly banner if the API is unreachable (CORS is enabled on the server).

## 3. Design

Calm, technical, trustworthy, like an industrial dashboard, not a marketing page. Light theme with a dark navy sidebar/header (`#0B1F33`), neutral grays, one teal accent (`#0E9AA7`). Font: Inter. Cards with soft shadows and 12 px radius; generous spacing; clear hierarchy.
**Status chips (use consistently):** `assumed` = amber, `detected` = green, `needs review` = red, `placeholder image` = gray, `expected position` = blue outline, `E57` / `IMAGE` method badges.
Do not copy any company's logo or branding. App name: **AutoTag360**.
Header: logo text, method switcher (A / B), Settings. Empty, loading (skeletons) and error states for every view.

## 4. Screens

### 4.1 Home / method chooser

Two large cards. Call `GET /methods` and show live facts on each card:

- **E57 scan**: "N cabinets found of M known", button **Show cabinets**. Disabled (with reason) if `e57.available` is false.
- **Image + coordinates**: "N photos available", button **Open photo viewer**. If `image.available` is false show "photos not available on the server".

### 4.2 Method A: E57 results (`GET /methods/e57/tags`)

- Top bar: title, count, **Export** menu (see 4.5), and a yellow banner listing `missing` cabinets: "Not found by the E57 scan: TSK1" (these are cabinets whose label was not read; do not hide them).
- Grid of **cabinet cards**: picture (`tag.image`, with a gray "placeholder image" chip when `image_is_placeholder`), name, folder, panel chip (`panel_model` + `panel_source`, confidence if present), device count ("No assets detected yet" when empty: never fake devices), documents count, `needs_review` chip.
- Click a card -> **Tag drawer** (4.4). Search box + filter (all / with documents / needs review).

### 4.3 Method B: Image + coordinates

Two tabs on the left:

1. **Photos**: `GET /photos` as a thumbnail grid (`image_url?w=320`, name, category chip none/single/multiple, cabinet ids). Filters: category (all/none/single/multiple) and cabinet (`?cabinet=H02`). Selecting a photo runs `POST /locate {photo}`.
2. **Coordinates**: inputs **x, y, z** (metres) and the orientation as a quaternion **w, x, y, z**, an **Analyze** button, and a **Fill from photo** helper (copies the selected photo's pose from `GET /photos`). Runs `POST /locate {position, rotation_wxyz}`. If no photo is selected there is no photo to show: display a neutral "No image for a typed position" panel and the results list.

**Result area (after /locate):**

- If `category === "none"`: a large empty state "No cabinet in view. No tag is created for this position." plus the API `message`. Export buttons disabled.
- Otherwise: the **photo viewer** (`image_url`, width 1280) with an SVG overlay. Boxes come in **photo pixels (frame_px = 4096)**: scale by `shownWidth / frame_px`. For each cabinet in `cabinets[].assets` draw: panel = cyan outline, relay (`abb_relion_615`) = yellow, VD4 window = green, look-alike display (`other_hmi`) = gray dashed. Assets have `status: "expected"`: draw them **dashed** with the label "expected position (geometry)". Draw `detected_devices[].xyxy` as **solid** boxes labelled "detected" with confidence. A legend, hover tooltip (label, class), and click-to-select a cabinet (syncs with the right panel).
- Right panel: one **cabinet card** per entry of `cabinets[]` (same card as 4.2, plus `view.depth_m` and `view.view_angle_deg`). Show **two pictures** per cabinet: the stored picture (`tag.image`) and **"From this photo"**, a crop of the photo around the panel box (use the panel asset's `xyxy` on the 1280-px image, with a canvas or CSS crop).
- Message line (`message`), and **Export** menu (4.5) for this photo/pose.

### 4.4 Tag drawer (both methods)

Large picture (with placeholder chip), **name**, folder, position `{x,y,z}` (E57 coordinates; "—" if null), panel chip, confidence, `needs_review`.
**Devices** list: type, class, confidence, review reasons, its own documents (a relay carries the REX615 manual, a VD4 window the VD4 manual); empty -> "No assets detected yet".
**Documents** list: title, model, pages, **Open** button -> PDF viewer modal (`<iframe src="{base}/documents/{id}/file#page=1">`) with a "open in new tab" link. Refresh with `GET /tags/{id}` when opened.
Optional: **Replace picture** (file input -> `PUT /tags/{id}/image?placeholder=false`).
**Ask the manuals** section: a disabled input with the text "Available after the assets of this cabinet are detected". **Do not call `/ask` yet.**

### 4.5 Export menu (both methods)

Buttons that call the backend with the current selection (`method: "e57"`, or `method: "image"` with `photo` or `position` + `rotation_wxyz`):

- **Manual tagging sheet** (`POST /export/manual`): JSON preview + **Download CSV** (`format: "csv"`, download the response as a file) + Download JSON. Explain in one line: "What a person creates by hand in the digital twin: name, position, picture and documents per cabinet." Render `instructions` as a short checklist and `rows` as a table.
- **Matterport JSON** (`POST /export/matterport`) with a toggle `model_api` | `sdk`: show formatted JSON (monospace, copy button, download button), the `skipped` list (cabinets without a position), and the `notes` array in an info box. Always show: "**Dry run: nothing is sent to Matterport.** Coordinates are E57 coordinates, not yet transformed to the Matterport model."
  For the image method with no cabinet in view the exports return an empty list: show "Nothing to export".

## 5. API reference (use exactly these; base URL = Settings)

| Call                                                                        | Use                                                                                                             |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `GET /health`                                                               | connection check                                                                                                |
| `GET /methods`                                                              | `{e57:{available,cabinets_found,cabinets_known,description}, image:{available,photos,manual_pose,description}}` |
| `GET /methods/e57/tags`                                                     | `{method,site,count,missing:[ids],note,tags:[Tag]}`                                                             |
| `GET /photos?category=&cabinet=`                                            | `{count,photos:[{name,scan,position:{x,y,z},rotation_wxyz:[w,x,y,z],category:"none                              | single          | multiple",cabinets:[ids],image_url}]}`; 503 if photos are not on the server |
| `GET /photos/{name}/image?w=1280`                                           | JPEG                                                                                                            |
| `POST /locate` body `{photo?, position?:{x,y,z}, rotation_wxyz?:[w,x,y,z]}` | below                                                                                                           |
| `GET /tags/{id}`                                                            | one Tag                                                                                                         |
| `GET /documents/{id}/file`                                                  | PDF (`#page=N` opens a page)                                                                                    |
| `POST /export/manual` body `{method:"e57"                                   | "image", photo?, position?, rotation_wxyz?, format?:"json"                                                      | "csv"}`         | manual tagging sheet (CSV is a file download)                               |
| `POST /export/matterport` body same + `format:"model_api"                   | "sdk"`                                                                                                          | Matterport JSON |
| `PUT /tags/{id}/image?placeholder=false` (multipart `file`)                 | optional: replace a picture                                                                                     |
| `POST /ask`                                                                 | **not used yet**                                                                                                |

**Tag** = `{id,name,folder,position:{x,y,z}|null,confidence,needs_review,image:"/assets/context/H05.jpg"|null,image_is_placeholder,found_by_e57,panel_model,panel_source:"assumed|detected"|null,panel_confidence,documents:[Doc],devices:[Device]}`
**Doc** = `{id,title,model,doc_type,pages,url:"/documents/2/file"}`
**Device** = `{id,tag_id,type,class,model,confidence,needs_review,review_reasons:[],position,ocr_text,crop,documents:[Doc],source:{image,box:[x,y,w,h]}}`
**/locate response** = `{method:"image",photo,image_url,pose:{position,rotation_wxyz},category,message,frame_px:4096,cabinets:[{tag:Tag,view:{cabinet,u,v,depth_m,view_angle_deg},assets:[{class,label,kind:"panel|device|lookalike",xyxy:[x0,y0,x1,y1],visible_fraction,status:"expected"}],detected_devices:[Device + xyxy]}],note}`

## 6. Rules and acceptance checks

- Never show a fake tag, device, detection, confidence, or position. Unknown = "—". `assumed` panels stay amber until `panel_source` is `detected`.
- "Expected" boxes are geometry predictions, not detections: label and style them differently from detected devices.
- Method B with a photo that shows no cabinet (e.g. `img_001.jpg`) must show the empty state and disable exports.
- Method A must list 9 cabinets and the banner "Not found by the E57 scan: TSK1".
- `img_033.jpg` (multiple) shows four cabinets (H01-H04); `img_067.jpg` (single) shows H02 with panel, relay and VD4-window boxes.
- PDFs open from `/documents/{id}/file`; the Matterport export for E57 contains 9 tags; the manual sheet CSV downloads as a file.
- Keyboard accessible, readable contrast, no horizontal scroll at 1024 px.
