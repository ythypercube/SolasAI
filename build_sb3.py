#!/usr/bin/env python3
"""
Build SolasGPT.sb3 – TurboWarp AI chat project.
Uses the gsaHTTPRequests (HTTP) extension to speak to the Render backend.
"""
import json, zipfile, uuid
import os
import hashlib

# ─── Config ──────────────────────────────────────────────────────────────────
API_KEY_PLACEHOLDER = os.getenv("SOLAS_API_KEY_PLACEHOLDER", "REPLACE_WITH_YOUR_RENDER_API_KEY")
BACKEND_URL = os.getenv("SOLAS_BACKEND_URL", "https://solasai-backend.onrender.com/chat-plain")
FEEDBACK_URL = os.getenv("SOLAS_FEEDBACK_URL", "https://solasai-backend.onrender.com/feedback")
OUT_PATH = os.getenv("SOLAS_SB3_OUT", "/mnt/data/SolasAI/turbowarp-ai-backend/SolasGPT.sb3")
INCLUDE_API_KEY_HEADER = os.getenv("SOLAS_INCLUDE_API_KEY_HEADER", "false").lower() == "true"
ROBOT_BRIDGE_URL = os.getenv("SOLAS_ROBOT_BRIDGE_URL", "http://localhost:8900")

# ─── ID helpers ──────────────────────────────────────────────────────────────
_counter = [0]
def uid(prefix="b"):
    _counter[0] += 1
    return f"{prefix}_{_counter[0]:04d}"

# ─── Variable / list IDs ─────────────────────────────────────────────────────
V = {k: uid("v") for k in ["sessionId", "userPrompt", "assistantReply", "statusText", "lineIndex", "responseIndex", "feedbackRating", "feedbackImprove", "awaitingFeedback", "shouldRetry", "showChatLog", "robotCmd"]}
L = {"chatLog": uid("l")}

# ─── Block factory ───────────────────────────────────────────────────────────
blocks: dict = {}

def B(bid, opcode, *, nxt=None, par=None, inp=None, fld=None,
      shadow=False, top=False, x=0, y=0):
    b = {"opcode": opcode, "next": nxt, "parent": par,
         "inputs": inp or {}, "fields": fld or {},
         "shadow": shadow, "topLevel": top}
    if top:
        b["x"] = x
        b["y"] = y
    blocks[bid] = b
    return bid

def lit(v):
    """Shadow literal string input."""
    return [1, [10, str(v)]]

def bi(bid, default=""):
    """Reporter block used as input."""
    return [3, bid, [10, str(default)]]

def vreporter(bid, vname, vid, par):
    """Create a data_variable reporter block and return a bi() reference."""
    B(bid, "data_variable", par=par, fld={"VARIABLE": [vname, vid]})
    return bi(bid, "")

# ─── GREEN FLAG SCRIPT ────────────────────────────────────────────────────────
gf = B(uid(), "event_whenflagclicked", top=True, x=50, y=50)

# set sessionId to join("user-", pick random 1000 to 9999)
gf_rnd  = B(uid(), "operator_random",
            inp={"FROM": lit("1000"), "TO": lit("9999")})
gf_join = B(uid(), "operator_join",
            inp={"STRING1": lit("user-"), "STRING2": bi(gf_rnd, "0")})
gf_set1 = B(uid(), "data_setvariableto",
            inp={"VALUE": bi(gf_join, "")},
            fld={"VARIABLE": ["sessionId", V["sessionId"]]})

# delete all of chatLog
gf_del = B(uid(), "data_deletealloflist",
           fld={"LIST": ["chatLog", L["chatLog"]]})

# set statusText to "Ready"
gf_sts = B(uid(), "data_setvariableto",
           inp={"VALUE": lit("Ready")},
           fld={"VARIABLE": ["statusText", V["statusText"]]})

# Wire green flag chain
blocks[gf]["next"]     = gf_set1
blocks[gf_rnd]["parent"]  = gf_join
blocks[gf_join]["parent"] = gf_set1
blocks[gf_set1]["parent"] = gf
blocks[gf_set1]["next"]   = gf_del
blocks[gf_del]["parent"]  = gf_set1
blocks[gf_del]["next"]    = gf_sts
blocks[gf_sts]["parent"]  = gf_del

ga = B(uid(), "event_whenflagclicked", top=True, x=850, y=50)

ga_loop = B(uid(), "control_forever", inp={})
blocks[ga]["next"] = ga_loop
blocks[ga_loop]["parent"] = ga

ga_status_1 = B(uid(), "data_variable", fld={"VARIABLE": ["statusText", V["statusText"]]})
ga_is_thinking_1 = B(uid(), "operator_equals",
                     inp={"OPERAND1": bi(ga_status_1, ""), "OPERAND2": lit("Thinking...")})
ga_resp_1 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_has_idx_1 = B(uid(), "operator_gt",
                 inp={"OPERAND1": bi(ga_resp_1, "0"), "OPERAND2": lit("0")})
ga_can_anim_1 = B(uid(), "operator_and",
                  inp={"OPERAND1": [2, ga_is_thinking_1], "OPERAND2": [2, ga_has_idx_1]})
ga_idx_ref_1 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_set_dot_1 = B(uid(), "data_replaceitemoflist",
                 inp={"INDEX": bi(ga_idx_ref_1, "1"), "ITEM": lit("Generating Response.")},
                 fld={"LIST": ["chatLog", L["chatLog"]]})
ga_if_1 = B(uid(), "control_if", inp={"CONDITION": [2, ga_can_anim_1], "SUBSTACK": [2, ga_set_dot_1]})
ga_wait_1 = B(uid(), "control_wait", inp={"DURATION": lit("1")})

ga_status_2 = B(uid(), "data_variable", fld={"VARIABLE": ["statusText", V["statusText"]]})
ga_is_thinking_2 = B(uid(), "operator_equals",
                     inp={"OPERAND1": bi(ga_status_2, ""), "OPERAND2": lit("Thinking...")})
ga_resp_2 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_has_idx_2 = B(uid(), "operator_gt",
                 inp={"OPERAND1": bi(ga_resp_2, "0"), "OPERAND2": lit("0")})
ga_can_anim_2 = B(uid(), "operator_and",
                  inp={"OPERAND1": [2, ga_is_thinking_2], "OPERAND2": [2, ga_has_idx_2]})
ga_idx_ref_2 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_set_dot_2 = B(uid(), "data_replaceitemoflist",
                 inp={"INDEX": bi(ga_idx_ref_2, "1"), "ITEM": lit("Generating Response..")},
                 fld={"LIST": ["chatLog", L["chatLog"]]})
ga_if_2 = B(uid(), "control_if", inp={"CONDITION": [2, ga_can_anim_2], "SUBSTACK": [2, ga_set_dot_2]})
ga_wait_2 = B(uid(), "control_wait", inp={"DURATION": lit("1")})

ga_status_3 = B(uid(), "data_variable", fld={"VARIABLE": ["statusText", V["statusText"]]})
ga_is_thinking_3 = B(uid(), "operator_equals",
                     inp={"OPERAND1": bi(ga_status_3, ""), "OPERAND2": lit("Thinking...")})
ga_resp_3 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_has_idx_3 = B(uid(), "operator_gt",
                 inp={"OPERAND1": bi(ga_resp_3, "0"), "OPERAND2": lit("0")})
ga_can_anim_3 = B(uid(), "operator_and",
                  inp={"OPERAND1": [2, ga_is_thinking_3], "OPERAND2": [2, ga_has_idx_3]})
ga_idx_ref_3 = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
ga_set_dot_3 = B(uid(), "data_replaceitemoflist",
                 inp={"INDEX": bi(ga_idx_ref_3, "1"), "ITEM": lit("Generating Response...")},
                 fld={"LIST": ["chatLog", L["chatLog"]]})
ga_if_3 = B(uid(), "control_if", inp={"CONDITION": [2, ga_can_anim_3], "SUBSTACK": [2, ga_set_dot_3]})
ga_wait_3 = B(uid(), "control_wait", inp={"DURATION": lit("1")})

blocks[ga_loop]["inputs"] = {"SUBSTACK": [2, ga_if_1]}
blocks[ga_if_1]["parent"] = ga_loop
blocks[ga_if_1]["next"] = ga_wait_1
blocks[ga_wait_1]["parent"] = ga_if_1
blocks[ga_wait_1]["next"] = ga_if_2
blocks[ga_if_2]["parent"] = ga_wait_1
blocks[ga_if_2]["next"] = ga_wait_2
blocks[ga_wait_2]["parent"] = ga_if_2
blocks[ga_wait_2]["next"] = ga_if_3
blocks[ga_if_3]["parent"] = ga_wait_2
blocks[ga_if_3]["next"] = ga_wait_3
blocks[ga_wait_3]["parent"] = ga_if_3

sm = B(uid(), "event_whenflagclicked", top=True, x=450, y=50)

loop = B(uid(), "control_forever", inp={})
blocks[sm]["next"] = loop
blocks[loop]["parent"] = sm

# ask and wait
ask = B(uid(), "sensing_askandwait",
        inp={"QUESTION": lit("Type your message:")})

# set userPrompt to answer
ans_rep = B(uid(), "sensing_answer")
set_prompt = B(uid(), "data_setvariableto",
               inp={"VALUE": bi(ans_rep, "")},
               fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})

# if length(userPrompt) = 0 → stop this script
pr_r1    = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
len_blk  = B(uid(), "operator_length", inp={"STRING": bi(pr_r1, "")})
eq_blk   = B(uid(), "operator_equals",
             inp={"OPERAND1": bi(len_blk, "0"), "OPERAND2": lit("0")})
stop_blk = B(uid(), "control_stop",
             fld={"STOP_OPTION": ["this script", None]},
             inp={})
blocks[stop_blk]["mutation"] = {
    "tagName": "mutation", "children": [], "hasnext": "false"
}
if_blk = B(uid(), "control_if",
           inp={"CONDITION": [2, eq_blk], "SUBSTACK": [2, stop_blk]})

# add "You: " + userPrompt to chatLog
pr_r2      = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
join_you   = B(uid(), "operator_join",
               inp={"STRING1": lit("You: "), "STRING2": bi(pr_r2, "")})
add_you    = B(uid(), "data_addtolist",
               inp={"ITEM": bi(join_you, "")},
               fld={"LIST": ["chatLog", L["chatLog"]]})

# add "Generating Response" placeholder, then remember its index
add_generating = B(uid(), "data_addtolist",
                   inp={"ITEM": lit("Generating Response")},
                   fld={"LIST": ["chatLog", L["chatLog"]]})
len_chatlog_gen = B(uid(), "data_lengthoflist",
                    fld={"LIST": ["chatLog", L["chatLog"]]})
set_gen_idx = B(uid(), "data_setvariableto",
                inp={"VALUE": bi(len_chatlog_gen, "0")},
                fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})

# set statusText to "Thinking..."
set_think  = B(uid(), "data_setvariableto",
               inp={"VALUE": lit("Thinking...")},
               fld={"VARIABLE": ["statusText", V["statusText"]]})

# HTTP extension blocks
clr  = B(uid(), "gsaHTTPRequests_clearAll")

mth_shadow = B(uid(), "gsaHTTPRequests_menu_method", shadow=True,
               fld={"method": ["POST", None]})
set_method = B(uid(), "gsaHTTPRequests_setRequestmethod",
               inp={"method": [1, mth_shadow]})

mime_shadow = B(uid(), "gsaHTTPRequests_menu_mimeType", shadow=True,
                fld={"type": ["application/json", None]})
set_ctype   = B(uid(), "gsaHTTPRequests_setMimeType",
                inp={"type": [1, mime_shadow]})

set_hdr = None
if INCLUDE_API_KEY_HEADER:
    set_hdr = B(uid(), "gsaHTTPRequests_setHeaderData",
                inp={"name": lit("x-api-key"), "value": lit(API_KEY_PLACEHOLDER)})

# Build body: {"sessionId":"<sid>","message":"<prompt>"}
sid_r  = B(uid(), "data_variable", fld={"VARIABLE": ["sessionId", V["sessionId"]]})
pr_r3  = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
bj4    = B(uid(), "operator_join",
           inp={"STRING1": bi(pr_r3, ""), "STRING2": lit('"}')} )
bj3    = B(uid(), "operator_join",
           inp={"STRING1": lit('","message":"'), "STRING2": bi(bj4, "")})
bj2    = B(uid(), "operator_join",
           inp={"STRING1": bi(sid_r, ""), "STRING2": bi(bj3, "")})
bj1    = B(uid(), "operator_join",
           inp={"STRING1": lit('{"sessionId":"'), "STRING2": bi(bj2, "")})
set_body = B(uid(), "gsaHTTPRequests_setBody", inp={"text": bi(bj1, "")})

# send request
send_req = B(uid(), "gsaHTTPRequests_sendRequest", inp={"url": lit(BACKEND_URL)})

# set assistantReply to response
res_rep    = B(uid(), "gsaHTTPRequests_resData")
set_reply  = B(uid(), "data_setvariableto",
               inp={"VALUE": bi(res_rep, "")},
               fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})

# Error handling: empty response or ERROR: prefix -> server starting list item
# ── Robot-command detection ───────────────────────────────────────────────────
# If assistantReply contains "|||ROBOT_CMD|||", split it:
#   - store command string in robotCmd
#   - overwrite assistantReply with the human-readable display part
#   - POST the commands to the local robot bridge
ROBOT_SEP = "|||ROBOT_CMD|||"

rp_robot_contains = B(uid(), "data_variable",
                      fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
robot_contains = B(uid(), "operator_contains",
                   inp={"STRING1": bi(rp_robot_contains, ""), "STRING2": lit(ROBOT_SEP)})

# Step 1: extract command part (item 2) → robotCmd
rp_robot_cmd_src = B(uid(), "data_variable",
                     fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_robot_cmd = B(uid(), "strings_split",
                    inp={"ITEM": lit("2"), "STRING": bi(rp_robot_cmd_src, ""), "SPLIT": lit(ROBOT_SEP)})
set_robot_cmd_var = B(uid(), "data_setvariableto",
                      inp={"VALUE": bi(split_robot_cmd, "")},
                      fld={"VARIABLE": ["robotCmd", V["robotCmd"]]})

# Step 2: extract display part (item 1) → overwrite assistantReply
rp_robot_disp_src = B(uid(), "data_variable",
                      fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_robot_disp = B(uid(), "strings_split",
                     inp={"ITEM": lit("1"), "STRING": bi(rp_robot_disp_src, ""), "SPLIT": lit(ROBOT_SEP)})
set_reply_disp_only = B(uid(), "data_setvariableto",
                        inp={"VALUE": bi(split_robot_disp, "")},
                        fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})

# Step 3: POST commands to robot bridge (localhost)
clr_robot   = B(uid(), "gsaHTTPRequests_clearAll")
rmth_shadow = B(uid(), "gsaHTTPRequests_menu_method", shadow=True,
                fld={"method": ["POST", None]})
set_robot_method = B(uid(), "gsaHTTPRequests_setRequestmethod",
                     inp={"method": [1, rmth_shadow]})
rmime_shadow = B(uid(), "gsaHTTPRequests_menu_mimeType", shadow=True,
                 fld={"type": ["application/json", None]})
set_robot_ctype = B(uid(), "gsaHTTPRequests_setMimeType",
                    inp={"type": [1, rmime_shadow]})
rcmd_v  = B(uid(), "data_variable", fld={"VARIABLE": ["robotCmd", V["robotCmd"]]})
rbj2    = B(uid(), "operator_join", inp={"STRING1": bi(rcmd_v, ""), "STRING2": lit('"}')} )
rbj1    = B(uid(), "operator_join",
            inp={"STRING1": lit('{"commands":"'), "STRING2": bi(rbj2, "")})
set_robot_body = B(uid(), "gsaHTTPRequests_setBody", inp={"text": bi(rbj1, "")})
send_robot_req = B(uid(), "gsaHTTPRequests_sendRequest",
                   inp={"url": lit(ROBOT_BRIDGE_URL + "/execute")})

# Link the robot sub-chain: set_robot_cmd_var → set_reply_disp_only → clr_robot → ...
robot_subchain = [
    set_robot_cmd_var, set_reply_disp_only,
    clr_robot, set_robot_method, set_robot_ctype, set_robot_body, send_robot_req
]
for _ri, _bid in enumerate(robot_subchain):
    if _ri > 0:
        blocks[_bid]["parent"] = robot_subchain[_ri - 1]
    if _ri < len(robot_subchain) - 1:
        blocks[_bid]["next"] = robot_subchain[_ri + 1]

# Outer if block: only runs the sub-chain when reply contains the separator
robot_detect_if = B(uid(), "control_if",
                    inp={"CONDITION": [2, robot_contains],
                         "SUBSTACK":  [2, set_robot_cmd_var]})

# Error handling: empty response or ERROR: prefix -> server starting list item
rp_err1 = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
len_err = B(uid(), "operator_length", inp={"STRING": bi(rp_err1, "")})
eq_err_empty = B(uid(), "operator_equals",
                 inp={"OPERAND1": bi(len_err, "0"), "OPERAND2": lit("0")})

rp_err2 = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_err_prefix = B(uid(), "strings_split",
                     inp={"ITEM": lit("1"), "STRING": bi(rp_err2, ""), "SPLIT": lit("ERROR:")})
eq_err_prefix = B(uid(), "operator_equals",
                  inp={"OPERAND1": bi(split_err_prefix, ""), "OPERAND2": lit("")})
or_err = B(uid(), "operator_or", inp={"OPERAND1": [2, eq_err_empty], "OPERAND2": [2, eq_err_prefix]})

add_server_msg = B(uid(), "data_addtolist",
                   inp={"ITEM": lit("Server: The API is starting, please wait.")},
                   fld={"LIST": ["chatLog", L["chatLog"]]})

# Image response protocol: IMAGE_URL:<url>||MESSAGE:<msg>
rp_img = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_img_prefix = B(uid(), "strings_split",
                     inp={"ITEM": lit("1"), "STRING": bi(rp_img, ""), "SPLIT": lit("IMAGE_URL:")})
eq_img_prefix = B(uid(), "operator_equals",
                  inp={"OPERAND1": bi(split_img_prefix, ""), "OPERAND2": lit("")})

rp_img_url_a = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_img_payload = B(uid(), "strings_split",
                      inp={"ITEM": lit("2"), "STRING": bi(rp_img_url_a, ""), "SPLIT": lit("IMAGE_URL:")})
split_img_url = B(uid(), "strings_split",
                  inp={"ITEM": lit("1"), "STRING": bi(split_img_payload, ""), "SPLIT": lit("||MESSAGE:")})
add_ai_costume = B(uid(), "lmsAssets_addCostume",
                   inp={"URL": bi(split_img_url, ""), "NAME": lit("AI Image")})
switch_backdrop = B(uid(), "looks_switchbackdropto", inp={"BACKDROP": lit("AI Image")})

rp_img_msg = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_img_message = B(uid(), "strings_split",
                      inp={"ITEM": lit("2"), "STRING": bi(rp_img_msg, ""), "SPLIT": lit("||MESSAGE:")})
set_reply_to_image_message = B(uid(), "data_setvariableto",
                               inp={"VALUE": bi(split_img_message, "")},
                               fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})

blocks[add_ai_costume]["next"] = switch_backdrop
blocks[switch_backdrop]["parent"] = add_ai_costume
blocks[switch_backdrop]["next"] = set_reply_to_image_message
blocks[set_reply_to_image_message]["parent"] = switch_backdrop

# Split assistantReply by newline and add each line as separate list item
set_line_index = B(uid(), "data_setvariableto",
                   inp={"VALUE": lit("1")},
                   fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})

idx_cond = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_cond  = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_cond = B(uid(), "strings_split",
               inp={"ITEM": bi(idx_cond, "1"), "STRING": bi(rp_cond, ""), "SPLIT": lit("\n")})
eq_empty = B(uid(), "operator_equals",
             inp={"OPERAND1": bi(split_cond, ""), "OPERAND2": lit("")})

idx_first = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
eq_first_line = B(uid(), "operator_equals",
                  inp={"OPERAND1": bi(idx_first, "1"), "OPERAND2": lit("1")})

idx_add_first = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_add_first  = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_add_first = B(uid(), "strings_split",
                    inp={"ITEM": bi(idx_add_first, "1"), "STRING": bi(rp_add_first, ""), "SPLIT": lit("\n")})
join_ai_first = B(uid(), "operator_join",
                  inp={"STRING1": lit("SolasGPT: "), "STRING2": bi(split_add_first, "")})
add_ai_first = B(uid(), "data_addtolist",
                 inp={"ITEM": bi(join_ai_first, "")},
                 fld={"LIST": ["chatLog", L["chatLog"]]})

idx_add_next = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_add_next  = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_add_next = B(uid(), "strings_split",
                   inp={"ITEM": bi(idx_add_next, "1"), "STRING": bi(rp_add_next, ""), "SPLIT": lit("\n")})
add_ai_next = B(uid(), "data_addtolist",
                inp={"ITEM": bi(split_add_next, "")},
                fld={"LIST": ["chatLog", L["chatLog"]]})

if_first_line = B(uid(), "control_if_else",
                  inp={
                      "CONDITION": [2, eq_first_line],
                      "SUBSTACK": [2, add_ai_first],
                      "SUBSTACK2": [2, add_ai_next]
                  })

change_index = B(uid(), "data_changevariableby",
                 inp={"VALUE": lit("1")},
                 fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})

blocks[if_first_line]["next"] = change_index
blocks[change_index]["parent"] = if_first_line

repeat_until_lines = B(uid(), "control_repeat_until",
                       inp={"CONDITION": [2, eq_empty], "SUBSTACK": [2, if_first_line]})

blocks[set_line_index]["next"] = repeat_until_lines
blocks[repeat_until_lines]["parent"] = set_line_index

# Separate line-output flow for image replies
set_line_index_img = B(uid(), "data_setvariableto",
                       inp={"VALUE": lit("1")},
                       fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})

idx_cond_img = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_cond_img = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_cond_img = B(uid(), "strings_split",
                   inp={"ITEM": bi(idx_cond_img, "1"), "STRING": bi(rp_cond_img, ""), "SPLIT": lit("\n")})
eq_empty_img = B(uid(), "operator_equals",
                 inp={"OPERAND1": bi(split_cond_img, ""), "OPERAND2": lit("")})

idx_first_img = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
eq_first_line_img = B(uid(), "operator_equals",
                      inp={"OPERAND1": bi(idx_first_img, "1"), "OPERAND2": lit("1")})

idx_add_first_img = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_add_first_img = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_add_first_img = B(uid(), "strings_split",
                        inp={"ITEM": bi(idx_add_first_img, "1"), "STRING": bi(rp_add_first_img, ""), "SPLIT": lit("\n")})
join_ai_first_img = B(uid(), "operator_join",
                      inp={"STRING1": lit("SolasGPT: "), "STRING2": bi(split_add_first_img, "")})
add_ai_first_img = B(uid(), "data_addtolist",
                     inp={"ITEM": bi(join_ai_first_img, "")},
                     fld={"LIST": ["chatLog", L["chatLog"]]})

idx_add_next_img = B(uid(), "data_variable", fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})
rp_add_next_img = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
split_add_next_img = B(uid(), "strings_split",
                       inp={"ITEM": bi(idx_add_next_img, "1"), "STRING": bi(rp_add_next_img, ""), "SPLIT": lit("\n")})
add_ai_next_img = B(uid(), "data_addtolist",
                    inp={"ITEM": bi(split_add_next_img, "")},
                    fld={"LIST": ["chatLog", L["chatLog"]]})

if_first_line_img = B(uid(), "control_if_else",
                      inp={
                          "CONDITION": [2, eq_first_line_img],
                          "SUBSTACK": [2, add_ai_first_img],
                          "SUBSTACK2": [2, add_ai_next_img]
                      })

change_index_img = B(uid(), "data_changevariableby",
                     inp={"VALUE": lit("1")},
                     fld={"VARIABLE": ["lineIndex", V["lineIndex"]]})

blocks[if_first_line_img]["next"] = change_index_img
blocks[change_index_img]["parent"] = if_first_line_img

repeat_until_lines_img = B(uid(), "control_repeat_until",
                           inp={"CONDITION": [2, eq_empty_img], "SUBSTACK": [2, if_first_line_img]})

blocks[set_reply_to_image_message]["next"] = set_line_index_img
blocks[set_line_index_img]["parent"] = set_reply_to_image_message
blocks[set_line_index_img]["next"] = repeat_until_lines_img
blocks[repeat_until_lines_img]["parent"] = set_line_index_img

normal_text_flow = B(uid(), "control_if_else",
                     inp={
                         "CONDITION": [2, eq_img_prefix],
                         "SUBSTACK": [2, add_ai_costume],
                         "SUBSTACK2": [2, set_line_index]
                     })

server_or_normal = B(uid(), "control_if_else",
                     inp={
                         "CONDITION": [2, or_err],
                         "SUBSTACK": [2, add_server_msg],
                         "SUBSTACK2": [2, normal_text_flow]
                     })

# delete "Generating Response" placeholder now that real reply is in the list
gen_idx_ref = B(uid(), "data_variable", fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})
del_generating = B(uid(), "data_deleteoflist",
                   inp={"INDEX": bi(gen_idx_ref, "1")},
                   fld={"LIST": ["chatLog", L["chatLog"]]})
reset_gen_idx = B(uid(), "data_setvariableto",
                  inp={"VALUE": lit("0")},
                  fld={"VARIABLE": ["responseIndex", V["responseIndex"]]})

# Mandatory user feedback: must choose ✓ or ✗
set_fb_rating_empty = B(uid(), "data_setvariableto",
                        inp={"VALUE": lit("")},
                        fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
set_fb_improve_empty = B(uid(), "data_setvariableto",
                         inp={"VALUE": lit("")},
                         fld={"VARIABLE": ["feedbackImprove", V["feedbackImprove"]]})
set_await_on = B(uid(), "data_setvariableto",
                 inp={"VALUE": lit("1")},
                 fld={"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]})
set_rate_status = B(uid(), "data_setvariableto",
                    inp={"VALUE": lit("Rate response: click ✓ or ✗")},
                    fld={"VARIABLE": ["statusText", V["statusText"]]})

fb_r1 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_eq_good = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r1, ""), "OPERAND2": lit("✓")})
fb_r2 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_eq_tick = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r2, ""), "OPERAND2": lit("✅")})
fb_r3 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_eq_bad = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r3, ""), "OPERAND2": lit("✗")})
fb_r4 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_eq_cross = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r4, ""), "OPERAND2": lit("❌")})
fb_or_good = B(uid(), "operator_or", inp={"OPERAND1": [2, fb_eq_good], "OPERAND2": [2, fb_eq_tick]})
fb_or_bad = B(uid(), "operator_or", inp={"OPERAND1": [2, fb_eq_bad], "OPERAND2": [2, fb_eq_cross]})
fb_valid = B(uid(), "operator_or", inp={"OPERAND1": [2, fb_or_good], "OPERAND2": [2, fb_or_bad]})

wait_feedback_tick = B(uid(), "control_wait", inp={"DURATION": lit("0.2")})
repeat_until_feedback = B(uid(), "control_repeat_until",
                          inp={"CONDITION": [2, fb_valid], "SUBSTACK": [2, wait_feedback_tick]})
set_await_off = B(uid(), "data_setvariableto",
                  inp={"VALUE": lit("0")},
                  fld={"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]})

fb_r5 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_is_bad_1 = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r5, ""), "OPERAND2": lit("✗")})
fb_r6 = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_is_bad_2 = B(uid(), "operator_equals", inp={"OPERAND1": bi(fb_r6, ""), "OPERAND2": lit("❌")})
fb_is_bad = B(uid(), "operator_or", inp={"OPERAND1": [2, fb_is_bad_1], "OPERAND2": [2, fb_is_bad_2]})

# If ✗: ask what to improve, store it, then schedule retry
ask_improve = B(uid(), "sensing_askandwait",
                inp={"QUESTION": lit("What should I improve?")})
ans_improve = B(uid(), "sensing_answer")
set_improve = B(uid(), "data_setvariableto",
                inp={"VALUE": bi(ans_improve, "")},
                fld={"VARIABLE": ["feedbackImprove", V["feedbackImprove"]]})
set_shouldretry_1 = B(uid(), "data_setvariableto",
                      inp={"VALUE": lit("1")},
                      fld={"VARIABLE": ["shouldRetry", V["shouldRetry"]]})
blocks[ask_improve]["next"] = set_improve
blocks[set_improve]["parent"] = ask_improve
blocks[set_improve]["next"] = set_shouldretry_1
blocks[set_shouldretry_1]["parent"] = set_improve

set_shouldretry_0 = B(uid(), "data_setvariableto",
                      inp={"VALUE": lit("0")},
                      fld={"VARIABLE": ["shouldRetry", V["shouldRetry"]]})
set_retry_if_bad = B(uid(), "control_if_else",
                     inp={"CONDITION": [2, fb_is_bad],
                          "SUBSTACK":  [2, ask_improve],
                          "SUBSTACK2": [2, set_shouldretry_0]})

# set statusText to "Ready"
set_ready = B(uid(), "data_setvariableto",
              inp={"VALUE": lit("Ready")},
              fld={"VARIABLE": ["statusText", V["statusText"]]})
# Only show "Ready" when not retrying
retry_cond_ready = B(uid(), "data_variable",
                     fld={"VARIABLE": ["shouldRetry", V["shouldRetry"]]})
retry_eq_ready = B(uid(), "operator_equals",
                   inp={"OPERAND1": bi(retry_cond_ready, ""), "OPERAND2": lit("0")})
set_ready_if_good = B(uid(), "control_if",
                      inp={"CONDITION": [2, retry_eq_ready], "SUBSTACK": [2, set_ready]})

# Send feedback to backend
clr_fb = B(uid(), "gsaHTTPRequests_clearAll")
fb_mth_shadow = B(uid(), "gsaHTTPRequests_menu_method", shadow=True,
                  fld={"method": ["POST", None]})
set_fb_method = B(uid(), "gsaHTTPRequests_setRequestmethod",
                  inp={"method": [1, fb_mth_shadow]})
fb_mime_shadow = B(uid(), "gsaHTTPRequests_menu_mimeType", shadow=True,
                   fld={"type": ["application/json", None]})
set_fb_ctype = B(uid(), "gsaHTTPRequests_setMimeType",
                 inp={"type": [1, fb_mime_shadow]})

fb_sid = B(uid(), "data_variable", fld={"VARIABLE": ["sessionId", V["sessionId"]]})
fb_msg = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
fb_rep = B(uid(), "data_variable", fld={"VARIABLE": ["assistantReply", V["assistantReply"]]})
fb_rat = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackRating", V["feedbackRating"]]})
fb_imp = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackImprove", V["feedbackImprove"]]})

fb_j10 = B(uid(), "operator_join", inp={"STRING1": bi(fb_imp, ""), "STRING2": lit('"}')})
fb_j9 = B(uid(), "operator_join", inp={"STRING1": lit('","improvement":"'), "STRING2": bi(fb_j10, "")})
fb_j8 = B(uid(), "operator_join", inp={"STRING1": bi(fb_rat, ""), "STRING2": bi(fb_j9, "")})
fb_j7 = B(uid(), "operator_join", inp={"STRING1": lit('","rating":"'), "STRING2": bi(fb_j8, "")})
fb_j6 = B(uid(), "operator_join", inp={"STRING1": bi(fb_rep, ""), "STRING2": bi(fb_j7, "")})
fb_j5 = B(uid(), "operator_join", inp={"STRING1": lit('","reply":"'), "STRING2": bi(fb_j6, "")})
fb_j4 = B(uid(), "operator_join", inp={"STRING1": bi(fb_msg, ""), "STRING2": bi(fb_j5, "")})
fb_j3 = B(uid(), "operator_join", inp={"STRING1": lit('","message":"'), "STRING2": bi(fb_j4, "")})
fb_j2 = B(uid(), "operator_join", inp={"STRING1": bi(fb_sid, ""), "STRING2": bi(fb_j3, "")})
fb_j1 = B(uid(), "operator_join", inp={"STRING1": lit('{"sessionId":"'), "STRING2": bi(fb_j2, "")})
set_fb_body = B(uid(), "gsaHTTPRequests_setBody", inp={"text": bi(fb_j1, "")})
send_fb_req = B(uid(), "gsaHTTPRequests_sendRequest", inp={"url": lit(FEEDBACK_URL)})

# ─── Top-of-loop retry if/else: ask on fresh turn; retry with improvement appended ───
# Build modified prompt = original + " (Note: " + feedbackImprove + ")"
mod_imp_v    = B(uid(), "data_variable", fld={"VARIABLE": ["feedbackImprove", V["feedbackImprove"]]})
mod_j3       = B(uid(), "operator_join", inp={"STRING1": bi(mod_imp_v, ""), "STRING2": lit(")") })
mod_j2       = B(uid(), "operator_join", inp={"STRING1": lit(" (Note: "), "STRING2": bi(mod_j3, "")})
mod_prompt_v = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
mod_j1       = B(uid(), "operator_join", inp={"STRING1": bi(mod_prompt_v, ""), "STRING2": bi(mod_j2, "")})
set_modified_prompt = B(uid(), "data_setvariableto",
                        inp={"VALUE": bi(mod_j1, "")},
                        fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
clr_fi = B(uid(), "data_setvariableto",
           inp={"VALUE": lit("")},
           fld={"VARIABLE": ["feedbackImprove", V["feedbackImprove"]]})
del_all_chatlog = B(uid(), "data_deletealloflist",
                    fld={"LIST": ["chatLog", L["chatLog"]]})
pr_r2_retry = B(uid(), "data_variable", fld={"VARIABLE": ["userPrompt", V["userPrompt"]]})
join_you_retry = B(uid(), "operator_join",
                   inp={"STRING1": lit("You: "), "STRING2": bi(pr_r2_retry, "")})
add_you_retry = B(uid(), "data_addtolist",
                  inp={"ITEM": bi(join_you_retry, "")},
                  fld={"LIST": ["chatLog", L["chatLog"]]})
blocks[set_modified_prompt]["next"] = clr_fi
blocks[clr_fi]["parent"] = set_modified_prompt
blocks[clr_fi]["next"] = del_all_chatlog
blocks[del_all_chatlog]["parent"] = clr_fi
blocks[del_all_chatlog]["next"] = add_you_retry
blocks[add_you_retry]["parent"] = del_all_chatlog

# No-retry sub-chain: ask → set_prompt → if_blk → add_you
blocks[ask]["next"] = set_prompt
blocks[set_prompt]["parent"] = ask
blocks[set_prompt]["next"] = if_blk
blocks[if_blk]["parent"] = set_prompt
blocks[if_blk]["next"] = add_you
blocks[add_you]["parent"] = if_blk

shouldretry_cond = B(uid(), "data_variable",
                     fld={"VARIABLE": ["shouldRetry", V["shouldRetry"]]})
retry_eq_zero = B(uid(), "operator_equals",
                  inp={"OPERAND1": bi(shouldretry_cond, ""), "OPERAND2": lit("0")})
retry_ifelse = B(uid(), "control_if_else",
                 inp={"CONDITION": [2, retry_eq_zero],
                      "SUBSTACK":  [2, ask],
                      "SUBSTACK2": [2, set_modified_prompt]})

# ─── Wire send-message chain (next / parent) ─────────────────────────────────
chain = [retry_ifelse, add_generating, set_gen_idx, set_think,
         clr, set_method, set_ctype]
if set_hdr is not None:
    chain.append(set_hdr)
chain.extend([
    set_body, send_req, set_reply, robot_detect_if, server_or_normal, del_generating, reset_gen_idx,
    set_fb_rating_empty, set_fb_improve_empty, set_await_on, set_rate_status, repeat_until_feedback, set_await_off,
    set_retry_if_bad,
    clr_fb, set_fb_method, set_fb_ctype, set_fb_body, send_fb_req,
    set_ready_if_good
])

for i, bid in enumerate(chain):
    if i > 0:
        blocks[bid]["parent"] = chain[i - 1]
    if i < len(chain) - 1:
        blocks[bid]["next"] = chain[i + 1]

blocks[loop]["inputs"] = {"SUBSTACK": [2, retry_ifelse]}
blocks[retry_ifelse]["parent"] = loop

# Sub-block parents
for sub, par in [
    (ans_rep, set_prompt), (pr_r1, len_blk), (len_blk, eq_blk),
    (eq_blk, if_blk), (stop_blk, if_blk),
    (pr_r2, join_you), (join_you, add_you),
    (len_chatlog_gen, set_gen_idx),
    (gen_idx_ref, del_generating),
    (fb_r1, fb_eq_good), (fb_r2, fb_eq_tick), (fb_r3, fb_eq_bad), (fb_r4, fb_eq_cross),
    (fb_eq_good, fb_or_good), (fb_eq_tick, fb_or_good),
    (fb_eq_bad, fb_or_bad), (fb_eq_cross, fb_or_bad),
    (fb_or_good, fb_valid), (fb_or_bad, fb_valid),
    (fb_valid, repeat_until_feedback),
    (wait_feedback_tick, repeat_until_feedback),
    (fb_r5, fb_is_bad_1), (fb_r6, fb_is_bad_2), (fb_is_bad_1, fb_is_bad), (fb_is_bad_2, fb_is_bad),
    (fb_is_bad, set_retry_if_bad),
    (ask_improve, set_retry_if_bad), (set_shouldretry_0, set_retry_if_bad),
    (ans_improve, set_improve), (set_improve, set_shouldretry_1), (set_shouldretry_1, set_improve),
    (retry_cond_ready, retry_eq_ready), (retry_eq_ready, set_ready_if_good),
    (set_ready, set_ready_if_good),
    (shouldretry_cond, retry_eq_zero), (retry_eq_zero, retry_ifelse),
    (set_modified_prompt, retry_ifelse), (ask, retry_ifelse),
    (mod_imp_v, mod_j3), (mod_j3, mod_j2), (mod_j2, mod_j1),
    (mod_prompt_v, mod_j1), (mod_j1, set_modified_prompt),
    (pr_r2_retry, join_you_retry), (join_you_retry, add_you_retry),
    (fb_mth_shadow, set_fb_method), (fb_mime_shadow, set_fb_ctype),
    (fb_sid, fb_j2), (fb_msg, fb_j4), (fb_rep, fb_j6), (fb_rat, fb_j8), (fb_imp, fb_j10),
    (fb_j10, fb_j9), (fb_j9, fb_j8), (fb_j8, fb_j7), (fb_j7, fb_j6),
    (fb_j6, fb_j5), (fb_j5, fb_j4), (fb_j4, fb_j3), (fb_j3, fb_j2), (fb_j2, fb_j1), (fb_j1, set_fb_body),
    (ga_status_1, ga_is_thinking_1), (ga_is_thinking_1, ga_can_anim_1),
    (ga_resp_1, ga_has_idx_1), (ga_has_idx_1, ga_can_anim_1), (ga_can_anim_1, ga_if_1),
    (ga_idx_ref_1, ga_set_dot_1), (ga_set_dot_1, ga_if_1),
    (ga_status_2, ga_is_thinking_2), (ga_is_thinking_2, ga_can_anim_2),
    (ga_resp_2, ga_has_idx_2), (ga_has_idx_2, ga_can_anim_2), (ga_can_anim_2, ga_if_2),
    (ga_idx_ref_2, ga_set_dot_2), (ga_set_dot_2, ga_if_2),
    (ga_status_3, ga_is_thinking_3), (ga_is_thinking_3, ga_can_anim_3),
    (ga_resp_3, ga_has_idx_3), (ga_has_idx_3, ga_can_anim_3), (ga_can_anim_3, ga_if_3),
    (ga_idx_ref_3, ga_set_dot_3), (ga_set_dot_3, ga_if_3),
    (mth_shadow, set_method), (mime_shadow, set_ctype),
    (sid_r, bj2), (pr_r3, bj4),
    (bj4, bj3), (bj3, bj2), (bj2, bj1), (bj1, set_body),
    (res_rep, set_reply),
        (rp_robot_contains, robot_contains), (robot_contains, robot_detect_if),
        (set_robot_cmd_var, robot_detect_if),
        (rp_robot_cmd_src, split_robot_cmd), (split_robot_cmd, set_robot_cmd_var),
        (rp_robot_disp_src, split_robot_disp), (split_robot_disp, set_reply_disp_only),
        (rmth_shadow, set_robot_method), (rmime_shadow, set_robot_ctype),
        (rcmd_v, rbj2), (rbj2, rbj1), (rbj1, set_robot_body),
    (rp_err1, len_err), (len_err, eq_err_empty),
    (rp_err2, split_err_prefix), (split_err_prefix, eq_err_prefix),
    (eq_err_empty, or_err), (eq_err_prefix, or_err), (or_err, server_or_normal),
    (add_server_msg, server_or_normal), (normal_text_flow, server_or_normal),
    (rp_img, split_img_prefix), (split_img_prefix, eq_img_prefix), (eq_img_prefix, normal_text_flow),
    (add_ai_costume, normal_text_flow), (set_line_index, normal_text_flow),
    (rp_img_url_a, split_img_payload), (split_img_payload, split_img_url),
    (split_img_url, add_ai_costume),
    (rp_img_msg, split_img_message), (split_img_message, set_reply_to_image_message),
    (idx_cond, split_cond), (rp_cond, split_cond), (split_cond, eq_empty),
    (eq_empty, repeat_until_lines), (if_first_line, repeat_until_lines),
    (idx_first, eq_first_line), (eq_first_line, if_first_line),
    (add_ai_first, if_first_line), (add_ai_next, if_first_line),
    (idx_add_first, split_add_first), (rp_add_first, split_add_first),
    (split_add_first, join_ai_first), (join_ai_first, add_ai_first),
    (idx_add_next, split_add_next), (rp_add_next, split_add_next),
    (split_add_next, add_ai_next),
    (idx_cond_img, split_cond_img), (rp_cond_img, split_cond_img), (split_cond_img, eq_empty_img),
    (eq_empty_img, repeat_until_lines_img), (if_first_line_img, repeat_until_lines_img),
    (idx_first_img, eq_first_line_img), (eq_first_line_img, if_first_line_img),
    (add_ai_first_img, if_first_line_img), (add_ai_next_img, if_first_line_img),
    (idx_add_first_img, split_add_first_img), (rp_add_first_img, split_add_first_img),
    (split_add_first_img, join_ai_first_img), (join_ai_first_img, add_ai_first_img),
    (idx_add_next_img, split_add_next_img), (rp_add_next_img, split_add_next_img),
    (split_add_next_img, add_ai_next_img),
]:
    blocks[sub]["parent"] = par

BACKDROP_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360">'
    '<rect width="480" height="360" fill="#1e1e2e"/>'
    '<text x="240" y="60" font-family="Arial" font-size="28" font-weight="bold" '
    'fill="#cdd6f4" text-anchor="middle">SolasGPT</text>'
    '<text x="240" y="95" font-family="Arial" font-size="14" '
    'fill="#a6adc8" text-anchor="middle">Type in the prompt when asked</text>'
    '</svg>'
)

BACKDROP_ID = hashlib.md5(BACKDROP_SVG.encode("utf-8")).hexdigest()

TICK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
    '<circle cx="32" cy="32" r="30" fill="#22c55e"/>'
    '<path d="M18 34 L28 44 L46 22" stroke="#ffffff" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
X_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
    '<circle cx="32" cy="32" r="30" fill="#ef4444"/>'
    '<path d="M20 20 L44 44 M44 20 L20 44" stroke="#ffffff" stroke-width="7" fill="none" stroke-linecap="round"/>'
    '</svg>'
)

TICK_ID = hashlib.md5(TICK_SVG.encode("utf-8")).hexdigest()
X_ID = hashlib.md5(X_SVG.encode("utf-8")).hexdigest()

tick_blocks = {}
x_blocks = {}

tick_evt = uid("tb")
tick_wait_var = uid("tb")
tick_eq = uid("tb")
tick_if = uid("tb")
tick_set = uid("tb")
tick_flag = uid("tb")
tick_forever = uid("tb")
tick_vis_var = uid("tb")
tick_vis_eq = uid("tb")
tick_show = uid("tb")
tick_hide = uid("tb")
tick_vis_ifelse = uid("tb")
tick_blocks[tick_evt] = {
    "opcode": "event_whenthisspriteclicked", "next": tick_if, "parent": None,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 220, "y": 260
}
tick_blocks[tick_wait_var] = {
    "opcode": "data_variable", "next": None, "parent": tick_eq,
    "inputs": {}, "fields": {"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]},
    "shadow": False, "topLevel": False
}
tick_blocks[tick_eq] = {
    "opcode": "operator_equals", "next": None, "parent": tick_if,
    "inputs": {"OPERAND1": bi(tick_wait_var, "0"), "OPERAND2": lit("1")},
    "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_if] = {
    "opcode": "control_if", "next": None, "parent": tick_evt,
    "inputs": {"CONDITION": [2, tick_eq], "SUBSTACK": [2, tick_set]},
    "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_set] = {
    "opcode": "data_setvariableto", "next": None, "parent": tick_if,
    "inputs": {"VALUE": lit("✓")},
    "fields": {"VARIABLE": ["feedbackRating", V["feedbackRating"]]},
    "shadow": False, "topLevel": False
}
tick_blocks[tick_flag] = {
    "opcode": "event_whenflagclicked", "next": tick_forever, "parent": None,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 220, "y": 180
}
tick_blocks[tick_forever] = {
    "opcode": "control_forever", "next": None, "parent": tick_flag,
    "inputs": {"SUBSTACK": [2, tick_vis_ifelse]}, "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_vis_var] = {
    "opcode": "data_variable", "next": None, "parent": tick_vis_eq,
    "inputs": {}, "fields": {"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]},
    "shadow": False, "topLevel": False
}
tick_blocks[tick_vis_eq] = {
    "opcode": "operator_equals", "next": None, "parent": tick_vis_ifelse,
    "inputs": {"OPERAND1": bi(tick_vis_var, "0"), "OPERAND2": lit("1")},
    "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_show] = {
    "opcode": "looks_show", "next": None, "parent": tick_vis_ifelse,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_hide] = {
    "opcode": "looks_hide", "next": None, "parent": tick_vis_ifelse,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": False
}
tick_blocks[tick_vis_ifelse] = {
    "opcode": "control_if_else", "next": None, "parent": tick_forever,
    "inputs": {"CONDITION": [2, tick_vis_eq], "SUBSTACK": [2, tick_show], "SUBSTACK2": [2, tick_hide]},
    "fields": {}, "shadow": False, "topLevel": False
}

x_evt = uid("xb")
x_wait_var = uid("xb")
x_eq = uid("xb")
x_if = uid("xb")
x_set = uid("xb")
x_flag = uid("xb")
x_forever = uid("xb")
x_vis_var = uid("xb")
x_vis_eq = uid("xb")
x_show = uid("xb")
x_hide = uid("xb")
x_vis_ifelse = uid("xb")
x_blocks[x_evt] = {
    "opcode": "event_whenthisspriteclicked", "next": x_if, "parent": None,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 300, "y": 260
}
x_blocks[x_wait_var] = {
    "opcode": "data_variable", "next": None, "parent": x_eq,
    "inputs": {}, "fields": {"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]},
    "shadow": False, "topLevel": False
}
x_blocks[x_eq] = {
    "opcode": "operator_equals", "next": None, "parent": x_if,
    "inputs": {"OPERAND1": bi(x_wait_var, "0"), "OPERAND2": lit("1")},
    "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_if] = {
    "opcode": "control_if", "next": None, "parent": x_evt,
    "inputs": {"CONDITION": [2, x_eq], "SUBSTACK": [2, x_set]},
    "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_set] = {
    "opcode": "data_setvariableto", "next": None, "parent": x_if,
    "inputs": {"VALUE": lit("✗")},
    "fields": {"VARIABLE": ["feedbackRating", V["feedbackRating"]]},
    "shadow": False, "topLevel": False
}
x_blocks[x_flag] = {
    "opcode": "event_whenflagclicked", "next": x_forever, "parent": None,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 300, "y": 180
}
x_blocks[x_forever] = {
    "opcode": "control_forever", "next": None, "parent": x_flag,
    "inputs": {"SUBSTACK": [2, x_vis_ifelse]}, "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_vis_var] = {
    "opcode": "data_variable", "next": None, "parent": x_vis_eq,
    "inputs": {}, "fields": {"VARIABLE": ["awaitingFeedback", V["awaitingFeedback"]]},
    "shadow": False, "topLevel": False
}
x_blocks[x_vis_eq] = {
    "opcode": "operator_equals", "next": None, "parent": x_vis_ifelse,
    "inputs": {"OPERAND1": bi(x_vis_var, "0"), "OPERAND2": lit("1")},
    "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_show] = {
    "opcode": "looks_show", "next": None, "parent": x_vis_ifelse,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_hide] = {
    "opcode": "looks_hide", "next": None, "parent": x_vis_ifelse,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": False
}
x_blocks[x_vis_ifelse] = {
    "opcode": "control_if_else", "next": None, "parent": x_forever,
    "inputs": {"CONDITION": [2, x_vis_eq], "SUBSTACK": [2, x_show], "SUBSTACK2": [2, x_hide]},
    "fields": {}, "shadow": False, "topLevel": False
}

# ─── showChatLog visibility control loop ───────────────────────────────────────
cl_flag    = uid("cl")
cl_forever = uid("cl")
cl_var     = uid("cl")
cl_eq      = uid("cl")
cl_show    = uid("cl")
cl_hide    = uid("cl")
cl_ifelse  = uid("cl")
blocks[cl_flag] = {
    "opcode": "event_whenflagclicked", "next": cl_forever, "parent": None,
    "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 620, "y": 60
}
blocks[cl_forever] = {
    "opcode": "control_forever", "next": None, "parent": cl_flag,
    "inputs": {"SUBSTACK": [2, cl_ifelse]}, "fields": {}, "shadow": False, "topLevel": False
}
blocks[cl_var] = {
    "opcode": "data_variable", "next": None, "parent": cl_eq,
    "inputs": {}, "fields": {"VARIABLE": ["showChatLog", V["showChatLog"]]},
    "shadow": False, "topLevel": False
}
blocks[cl_eq] = {
    "opcode": "operator_equals", "next": None, "parent": cl_ifelse,
    "inputs": {"OPERAND1": bi(cl_var, ""), "OPERAND2": lit("1")},
    "fields": {}, "shadow": False, "topLevel": False
}
blocks[cl_show] = {
    "opcode": "data_showlist", "next": None, "parent": cl_ifelse,
    "inputs": {}, "fields": {"LIST": ["chatLog", L["chatLog"]]},
    "shadow": False, "topLevel": False
}
blocks[cl_hide] = {
    "opcode": "data_hidelist", "next": None, "parent": cl_ifelse,
    "inputs": {}, "fields": {"LIST": ["chatLog", L["chatLog"]]},
    "shadow": False, "topLevel": False
}
blocks[cl_ifelse] = {
    "opcode": "control_if_else", "next": None, "parent": cl_forever,
    "inputs": {"CONDITION": [2, cl_eq], "SUBSTACK": [2, cl_show], "SUBSTACK2": [2, cl_hide]},
    "fields": {}, "shadow": False, "topLevel": False
}
blocks[cl_forever]["parent"] = cl_flag
blocks[cl_ifelse]["parent"] = cl_forever
blocks[cl_show]["parent"] = cl_ifelse
blocks[cl_hide]["parent"] = cl_ifelse
blocks[cl_eq]["parent"] = cl_ifelse
blocks[cl_var]["parent"] = cl_eq

# ─── Project JSON ─────────────────────────────────────────────────────────────
project = {
    "targets": [
        {
            "isStage": True,
            "name": "Stage",
            "variables": {
                V["sessionId"]:      ["sessionId",      ""],
                V["userPrompt"]:     ["userPrompt",     ""],
                V["assistantReply"]: ["assistantReply", ""],
                V["statusText"]:     ["statusText",     ""],
                V["lineIndex"]:      ["lineIndex",      1],
                V["responseIndex"]:  ["responseIndex",  0],
                V["feedbackRating"]: ["feedbackRating", ""],
                V["feedbackImprove"]:["feedbackImprove", ""],
                V["awaitingFeedback"]:["awaitingFeedback", 0],
                V["shouldRetry"]:       ["shouldRetry", "0"],
                V["showChatLog"]:       ["showChatLog", 1],
                            V["robotCmd"]:          ["robotCmd", ""],
            },
            "lists": {
                L["chatLog"]: ["chatLog", []]
            },
            "broadcasts": {},
            "blocks": blocks,
            "comments": {},
            "currentCostume": 0,
            "costumes": [{
                "assetId": BACKDROP_ID,
                "name": "SolasGPT",
                "bitmapResolution": 1,
                "md5ext": f"{BACKDROP_ID}.svg",
                "dataFormat": "svg",
                "rotationCenterX": 240,
                "rotationCenterY": 180
            }],
            "sounds": [],
            "volume": 100,
            "layerOrder": 0,
            "tempo": 60,
            "videoTransparency": 50,
            "videoState": "on",
            "textToSpeechLanguage": None,
        },
        {
            "isStage": False,
            "name": "Tick",
            "variables": {},
            "lists": {},
            "broadcasts": {},
            "blocks": tick_blocks,
            "comments": {},
            "currentCostume": 0,
            "costumes": [{
                "assetId": TICK_ID,
                "name": "tick",
                "bitmapResolution": 1,
                "md5ext": f"{TICK_ID}.svg",
                "dataFormat": "svg",
                "rotationCenterX": 32,
                "rotationCenterY": 32
            }],
            "sounds": [],
            "volume": 100,
            "layerOrder": 1,
            "visible": False,
            "x": 222,
            "y": 148,
            "size": 60,
            "direction": 90,
            "draggable": False,
            "rotationStyle": "all around"
        },
        {
            "isStage": False,
            "name": "Cross",
            "variables": {},
            "lists": {},
            "broadcasts": {},
            "blocks": x_blocks,
            "comments": {},
            "currentCostume": 0,
            "costumes": [{
                "assetId": X_ID,
                "name": "cross",
                "bitmapResolution": 1,
                "md5ext": f"{X_ID}.svg",
                "dataFormat": "svg",
                "rotationCenterX": 32,
                "rotationCenterY": 32
            }],
            "sounds": [],
            "volume": 100,
            "layerOrder": 2,
            "visible": False,
            "x": 222,
            "y": 108,
            "size": 60,
            "direction": 90,
            "draggable": False,
            "rotationStyle": "all around"
        }
    ],
    "monitors": [
        {
            "id": L["chatLog"],
            "mode": "list",
            "opcode": "data_listcontents",
            "params": {"LIST": "chatLog"},
            "spriteName": None,
            "value": [],
            "width": 420,
            "height": 220,
            "x": 10,
            "y": 10,
            "visible": True,
        },
        {
            "id": V["statusText"],
            "mode": "default",
            "opcode": "data_variable",
            "params": {"VARIABLE": "statusText"},
            "spriteName": None,
            "value": "",
            "width": 0,
            "height": 0,
            "x": 10,
            "y": 350,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        },
        {
            "id": V["showChatLog"],
            "mode": "slider",
            "opcode": "data_variable",
            "params": {"VARIABLE": "showChatLog"},
            "spriteName": None,
            "value": 1,
            "width": 0,
            "height": 0,
            "x": 10,
            "y": 375,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 1,
            "isDiscrete": True,
        },
    ],
    "extensions": ["gsaHTTPRequests", "strings", "lmsAssets"],
    "extensionURLs": {
        "gsaHTTPRequests": "https://extensions.turbowarp.org/godslayerakp/http.js",
        "strings": "https://extensions.turbowarp.org/text.js",
        "lmsAssets": "https://extensions.turbowarp.org/Lily/Assets.js"
    },
    "meta": {
        "semver": "3.0.0",
        "vm": "2.3.0",
        "agent": "SolasAI project builder",
        "platform": {
            "name": "TurboWarp",
            "url": "https://turbowarp.org"
        }
    }
}

# ─── Write .sb3 ──────────────────────────────────────────────────────────────
with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("project.json",          json.dumps(project, separators=(",", ":")))
    zf.writestr(f"{BACKDROP_ID}.svg",    BACKDROP_SVG)
    zf.writestr(f"{TICK_ID}.svg",        TICK_SVG)
    zf.writestr(f"{X_ID}.svg",           X_SVG)

print(f"Built: {OUT_PATH}")
print(f"Blocks: {len(blocks)}")
print()
if INCLUDE_API_KEY_HEADER:
    print("IMPORTANT: Open SolasGPT.sb3 in TurboWarp, then:")
    print(f"  1. Find block: 'in request headers set [x-api-key] to [{API_KEY_PLACEHOLDER}]'")
    print("  2. Replace it with your real Render API key.")
    print("  3. Save and share the project.")
else:
    print("Project built without API key header.")
    print("Use this mode only when backend REQUIRE_API_KEY=false.")
