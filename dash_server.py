"""
Dash AI Agent Server
This server uses Flask and Socket.IO to handle real-time communication
with a web-based front-end. It integrates with the Ollama API to process
user inputs and provide intelligent responses related to driving assistance,
such as finding gas stations, restaurants, and providing route information.
"""

import json
from flask import Flask
from flask_socketio import SocketIO, emit
import ollama
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Set, Dict, Any

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here!'
socketio = SocketIO(app, cors_allowed_origins="*")

SHORT_TERM_MEMORY: Dict[str, List] = { "last_suggested_options": [] }

ROUTE_STATE = {
    "current_trip_state": {
        "origin": "Denver International Airport",
        "destination": "Estes Park",
        "total_eta_minutes": 105,
        "current_time": "5:00 PM",
        "final_eta_time": "6:45 PM",
        "stops_on_route": [
            {"name": "Broomfield", "eta_time": "5:35 PM", "eta_minutes": 35},
            {"name": "Boulder", "eta_time": "5:50 PM", "eta_minutes": 50},
            {"name": "Lyons", "eta_time": "6:15 PM", "eta_minutes": 75},
            {"name": "Estes Park", "eta_time": "6:45 PM", "eta_minutes": 105}
        ],
        "next_turn_index": 0,
        "turn_by_turn_directions": [
            {"instruction": "Take Exit 9 on the right.", "distance": "0.5 mi", "image_asset": "card_turn_exit_9.png"},
            {"instruction": "Turn right onto 1st Ave.", "distance": "2 mi", "image_asset": "card_turn_1st_ave.png"},
            {"instruction": "Stay in the left lane.", "distance": "5 mi", "image_asset": "card_turn_stay_left.png"},
            {"instruction": "Turn left onto East 57th St.", "distance": "1 mi", "image_asset": "card_turn_left_57.png"},
            {"instruction": "Arrive at Estes Park.", "distance": "", "image_asset": "card_turn_placeholder.png"}
        ],
        "live_data": {
            "current_speed_limit": "65 MPH",
            "traffic_ahead": "Light traffic reported, with a 5 minute delay near Boulder."
        }
    },
    
    # --- STRICT AMENITIES LIST (14 Items) ---
    "potential_amenities": [
        # GAS
        {"name": "Shell", "location": "Denver", "detour_minutes": 5, "asset_name": "card_shell.png", "stars": 4.3, "tags": ["gas", "chain", "reliable"]},
        {"name": "King Soopers Gas", "location": "Broomfield", "detour_minutes": 2, "asset_name": "card_king_soopers.png", "stars": 4.7, "tags": ["gas", "cheap", "good food selection"]},
        {"name": "Conoco", "location": "Boulder", "detour_minutes": 3, "asset_name": "card_conoco.png", "stars": 3.9, "tags": ["gas", "clean bathroom"]},
        {"name": "7-Eleven Gas", "location": "Broomfield", "detour_minutes": 3, "asset_name": "card_7_eleven.png", "stars": 4.1, "tags": ["gas", "cheap", "open late"]},
        {"name": "Exxon", "location": "Boulder", "detour_minutes": 1, "asset_name": "card_exxon.png", "stars": 4.8, "tags": ["gas", "reliable", "clean bathroom"]},
        {"name": "Circle K", "location": "Lyons", "detour_minutes": 1, "asset_name": "card_circle_k.png", "stars": 3.6, "tags": ["gas", "cheap"]},

        # FOOD
        {"name": "P.F. Chang’s", "location": "Broomfield", "detour_minutes": 7, "asset_name": "card_pf_changs.png", "stars": 4.4, "tags": ["food", "asian", "sit-down", "chain"]},
        {"name": "The Sink", "location": "Boulder", "detour_minutes": 8, "asset_name": "card_the_sink.png", "stars": 4.9, "tags": ["food", "burger", "historic", "local-favorite", "pizza"]},
        {"name": "Avery Brewing Co.", "location": "Boulder", "detour_minutes": 8, "asset_name": "card_avery_brewing.png", "stars": 4.5, "tags": ["food", "brewery", "bbq", "sit-down"]},
        {"name": "Smokin’ Dave’s BBQ", "location": "Lyons", "detour_minutes": 5, "asset_name": "card_smokin_daves_bbq.png", "stars": 4.0, "tags": ["food", "bbq", "sit-down", "local-favorite"]},
        {"name": "Panda Express", "location": "Broomfield", "detour_minutes": 4, "asset_name": "card_panda_express.png", "stars": 3.8, "tags": ["food", "asian", "fast-food", "chain"]},
        {"name": "Oskar Blues Grill & Brew", "location": "Lyons", "detour_minutes": 3, "asset_name": "card_oskar_blues.png", "stars": 4.7, "tags": ["food", "brewery", "burger", "sit-down"]},
        {"name": "Corner Bar", "location": "Boulder", "detour_minutes": 9, "asset_name": "card_corner_bar.png", "stars": 4.2, "tags": ["food", "bar", "sit-down", "casual"]},
        {"name": "Wayne’s Smoke Shack", "location": "Broomfield", "detour_minutes": 6, "asset_name": "card_waynes_smoke_shack.png", "stars": 4.8, "tags": ["food", "bbq", "unique", "local-favorite"]}
    ]
}

KNOWN_TAGS: Set[str] = {
    "gas", "clean bathroom", "cheap", "good food selection", "near highway", "chain", "reliable",
    "food", "sit-down", "asian", "good for groups", "burger", "historic", "unique", "local-favorite", "casual", "brewery", "modern-american", "bbq", "italian", "mexican", "thai", "nepalese", "indian", "fast-food", "bar", "pizza"
}

KNOWN_FOOD_VIBES: Set[str] = {
    "asian", "burger", "historic", "brewery", "bbq", "italian", "mexican", "pizza"
}


# --- 3. PYDANTIC MODELS ---
class ClassifiedIntent(BaseModel):
    intent: Literal[
        "find_gas",
        "find_food",
        "add_stop",
        "get_eta",
        "get_next_stop",
        "get_direction",
        "repeat_direction",
        "get_traffic",
        "get_speed_limit",
        "other"
    ] = Field(description="The user's primary goal.")
    
    tags: Optional[List[str]] = Field(default=None)
    location: Optional[str] = Field(default=None)
    stop_name: Optional[str] = Field(default=None)

class BestPick(BaseModel):
    best_match: Dict[str, Any] = Field(description="The single best JSON object from the provided 'options_to_rank'.")


# --- 4. SOCKET.IO EVENT HANDLERS ---
@socketio.on('connect')
def handle_connect():
    print("Client connected!")
    SHORT_TERM_MEMORY["last_suggested_options"] = []
    payload = {
        "response_type": "greeting",
        "text_to_speak": "I'm Dash, your driving assistant. How can I help?",
        "data": {}
    }
    emit('dash_response', payload)


@socketio.on('user_spoke')
def handle_user_spoke(json_data):
    user_text = json_data.get('text', '')
    if not user_text:
        return
    print(f"\n[USER]: \"{user_text}\"")
    payload = run_agent(user_text)
    print(f"[SERVER -> HTML]: {payload}")
    emit('dash_response', payload)


# --- 5. CORE AGENT LOGIC ---
def run_agent(user_text: str) -> dict:
    try:
        last_options = SHORT_TERM_MEMORY.get("last_suggested_options", [])
        intent_json = get_intent_from_ai(user_text, last_options)
        
        intent = ClassifiedIntent.model_validate(intent_json)
        print(f"[AI CLASSIFIER]: {intent.model_dump_json(indent=2)}")

        if intent.intent == "find_gas":
            return handle_find_gas(intent.tags, intent.location)
        elif intent.intent == "find_food":
            return handle_find_food(intent.tags, intent.location)
        elif intent.intent == "get_eta":
            return handle_get_eta(intent.location)
        elif intent.intent == "get_next_stop":
            return handle_get_next_stop()
        elif intent.intent == "get_direction":
            return handle_get_direction()
        elif intent.intent == "repeat_direction":
            return handle_get_direction(repeat=True)
        elif intent.intent == "get_traffic":
            return handle_get_traffic()
        elif intent.intent == "get_speed_limit":
            return handle_get_speed_limit()
        elif intent.intent == "add_stop":
            return handle_add_stop(intent.stop_name)
        else:
            return handle_other(user_text)

    except Exception as e:
        print(f"[AGENT ERROR]: {e}")
        return {
            "response_type": "error",
            "text_to_speak": "Sorry, I had a problem processing that.",
            "data": {}
        }


def get_intent_from_ai(user_text: str, memory: List[Dict[str, Any]]) -> dict:
    schema_string = json.dumps(ClassifiedIntent.model_json_schema(), indent=2)
    memory_context = "none"
    if memory:
        if "gas" in memory[0].get("tags", []):
            memory_context = "gas"
        elif "food" in memory[0].get("tags", []):
            memory_context = "food"

    system_prompt = f"""
    You are an intent classification engine. Your *only* job is to
    convert the user's text into a valid JSON object based on the
    Pydantic schema provided.
    
    Schema: {schema_string}
    CONTEXT: The user was just presented with '{memory_context}' options.
    
    RULES:
    - Ambiguous? Use CONTEXT.
    - "I need gas" -> {{"intent": "find_gas", "tags": ["gas"]}}
    - "I'm hungry" -> {{"intent": "find_food", "tags": null}}
    - "I want italian" -> {{"intent": "find_food", "tags": ["italian"]}}
    - "Yes", "Sure", "Add it" -> CLASSIFY AS: {{"intent": "add_stop", "stop_name": "that one"}}
    - "Add Shell" -> {{"intent": "add_stop", "stop_name": "Shell"}}
    
    User text: "{user_text}"
    """
    
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'system', 'content': system_prompt}],
            format='json',
            options={'temperature': 0.0}
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"intent": "other"}


def format_response_with_ai(context: dict, goal: str) -> str:
    context_json = json.dumps(context, indent=2)
    system_prompt = """
    You are 'Dash', a helpful, concise voice assistant.
    *** CRITICAL RULES ***
    1. ONLY talk about options explicitly listed in 'Data'.
    2. Do NOT invent new restaurant names.
    3. Speak naturally.
    """
    user_prompt = f"Goal: {goal}\n\nData:\n{context_json}"

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.1}
        )
        return response['message']['content'].strip()
    except Exception:
        return "Sorry, I had a problem generating a response."


def _extrapolate_best_option(options_list: List[Dict[str, Any]], unknown_tags: List[str]) -> Optional[Dict[str, Any]]:
    if not options_list: return None
    schema_string = json.dumps(BestPick.model_json_schema(), indent=2)
    system_prompt = f"""
    You are a data analyst. Find the SINGLE BEST match for request: {unknown_tags}.
    Return valid JSON only.
    Schema: {schema_string}
    """
    user_prompt = f"Data: {{ \"options_to_rank\": {json.dumps(options_list, indent=2)} }}"
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            format='json',
            options={'temperature': 0.0}
        )
        result = json.loads(response['message']['content'])
        return BestPick.model_validate(result).best_match
    except Exception:
        return None


def _build_option_speech(option: Dict[str, Any], justification_tags: List[str]) -> str:
    base = f"{option['name']} is a {option['detour_minutes']} minute detour"
    matching_tags = [t for t in option.get("tags", []) if t in justification_tags and t not in ("gas", "food")]
    if matching_tags:
        return f"{base}, recent reviews mention {', '.join(matching_tags)}."
    return f"{base}."

# --- 6. INTENT HANDLER FUNCTIONS ---

def handle_find_gas(tags: Optional[List[str]], location: Optional[str]) -> dict:
    print(f"Handling intent: find_gas (tags: {tags}, location: {location})")
    if not tags: tags = ["gas"]
    user_tags_set = set(t.lower() for t in tags)
    known_tags = user_tags_set.intersection(KNOWN_TAGS)
    unknown_tags = user_tags_set.difference(KNOWN_TAGS)
    
    matches = []
    for item in ROUTE_STATE['potential_amenities']:
        if "gas" in item['tags'] and all(tag in item['tags'] for tag in known_tags):
            if location and location.lower() not in item['location'].lower(): continue
            matches.append(item)

    if not matches:
        SHORT_TERM_MEMORY["last_suggested_options"] = []
        return {"response_type": "no_options", "text_to_speak": "Sorry, I couldn't find any gas stations matching that.", "data": {}}

    cards_to_show = []
    goal = ""

    if unknown_tags:
        # Use Analyst if tags are unknown
        best_option = _extrapolate_best_option(matches, list(unknown_tags))
        if not best_option: return handle_other("AI Analyst failed to pick an option.")
        context = {"user_request": list(unknown_tags), "best_option": best_option}
        # Explicit phrasing prompt
        goal = (
            "The user asked for gas with these qualities: {user_request}. "
            "You found that '{best_option['name']}' seems to be the best match. "
            "Justify this pick by saying **'recent reviews mention [Tag Match]'**. "
            "Also state its detour time. "
            "Then, ask the user to confirm if they want to add it."
        )
        cards_to_show = [best_option]
        text_response = format_response_with_ai(context, goal)
    else:
        # Standard Known Tag Match
        cards_to_show = matches[:3]
        justification_tags = [t for t in known_tags if t != "gas"]
        context = {"user_request_tags": justification_tags, "options_found": cards_to_show}
        
        if justification_tags:
            text_response = " || ".join(_build_option_speech(opt, justification_tags) for opt in cards_to_show)
        else:
            text_response = " || ".join(f"{opt['name']} is a {opt['detour_minutes']} minute detour." for opt in cards_to_show)

    SHORT_TERM_MEMORY["last_suggested_options"] = cards_to_show
    return {"response_type": "show_options", "text_to_speak": text_response, "data": { "cards_to_show": cards_to_show }}


def handle_find_food(tags: Optional[List[str]], location: Optional[str]) -> dict:
    print(f"Handling intent: find_food (tags: {tags}, location: {location})")
    
    if not tags or tags == ["food"]:
        vibes = sorted(list(KNOWN_FOOD_VIBES))
        context = {"available_vibes": vibes, "location": location}
        goal = (
            "The user is hungry" + (f" and asked about {location}." if location else ".") +
            " Ask them what kind of food they're in the mood for. "
            "Suggest 3-4 examples from the available list (e.g., Italian, Mexican, BBQ)."
        )
        text = format_response_with_ai(context, goal)
        SHORT_TERM_MEMORY["last_suggested_options"] = [] 
        return {"response_type": "clarification", "text_to_speak": text, "data": {"available_vibes": vibes}}

    user_tags_set = set(t.lower() for t in tags)
    known_tags_in_request = user_tags_set.intersection(KNOWN_TAGS)
    unknown_tags_in_request = user_tags_set.difference(KNOWN_TAGS)

    matches = []
    for item in ROUTE_STATE['potential_amenities']:
        if "food" in item['tags'] and all(tag in item['tags'] for tag in known_tags_in_request):
            if location and location.lower() not in item['location'].lower(): continue
            matches.append(item)

    if not matches and not unknown_tags_in_request:
        SHORT_TERM_MEMORY["last_suggested_options"] = []
        return handle_other(f"No food found matching {tags}.")

    cards_to_show = []
    goal = ""

    if unknown_tags_in_request:
        search_pool = matches if matches else [i for i in ROUTE_STATE['potential_amenities'] if "food" in i['tags']]
        best_option = _extrapolate_best_option(search_pool, list(unknown_tags_in_request))
        if not best_option: return handle_other("AI Analyst failed.")
        
        context = {"user_request_tags": list(unknown_tags_in_request), "best_option_found": best_option}
        goal = (
            "The user asked for food with these qualities: {user_request_tags}. "
            "You found that '{best_option_found[name]}' seems to be the best match. "
            "Justify this pick by **implying you looked at its data** (e.g., 'recent reviews mention {best_option_found[tags_...]}') and state its detour time. "
            "Then, **ask the user to confirm if they want to add it**."
        )
        cards_to_show = [best_option]
        text_response = format_response_with_ai(context, goal)

    else:
        cards_to_show = matches[:3]
        justification_tags = [t for t in known_tags_in_request if t != "food"]
        context = {"user_request_tags": justification_tags, "options_found": cards_to_show}
        
        if justification_tags:
            text_response = " || ".join(_build_option_speech(opt, justification_tags) for opt in cards_to_show)
        else:
            text_response = " || ".join(f"{opt['name']} is a {opt['detour_minutes']} minute detour." for opt in cards_to_show)

    SHORT_TERM_MEMORY["last_suggested_options"] = cards_to_show
    return {"response_type": "show_options", "text_to_speak": text_response, "data": { "cards_to_show": cards_to_show }}


def handle_add_stop(stop_name: Optional[str]) -> dict:
    target = None
    s_name = str(stop_name).lower() if stop_name else ""
    
    idx_to_add = -1
    if "second" in s_name or "2nd" in s_name: idx_to_add = 1
    elif "third" in s_name or "3rd" in s_name: idx_to_add = 2
    elif "first" in s_name or "1st" in s_name: idx_to_add = 0
    
    is_memory_ref = idx_to_add != -1 or not stop_name or any(x in s_name for x in ["that", "this", "it", "one", "option"])
    
    print(f"Adding stop. Name: {stop_name}, Memory Ref: {is_memory_ref}, Index: {idx_to_add}")

    if is_memory_ref and SHORT_TERM_MEMORY["last_suggested_options"]:
        options = SHORT_TERM_MEMORY["last_suggested_options"]
        if idx_to_add != -1:
            if idx_to_add < len(options):
                target = options[idx_to_add]
            else:
                return handle_other(f"You asked for the {stop_name}, but I only found {len(options)} options.")
        else:
            target = options[0]
            
    elif stop_name and not is_memory_ref:
        for am in ROUTE_STATE['potential_amenities']:
            if stop_name.lower() in am['name'].lower():
                target = am
                break
                
    if not target: return handle_other(f"Can't find stop {stop_name}")
    
    SHORT_TERM_MEMORY["last_suggested_options"] = []
    new_eta = ROUTE_STATE['current_trip_state']['total_eta_minutes'] + target['detour_minutes']
    text = format_response_with_ai(
        {"stop": target, "new_eta": new_eta}, 
        "Confirm that you are adding '{stop[name]}' to the route and state the new total ETA."
    )
    return {"response_type": "stop_added", "text_to_speak": text, "data": {"new_eta_minutes": new_eta}}

def handle_other(user_text: str) -> dict:
    print(f"Handling as 'other': {user_text}")
    context = {"user_text": user_text}
    text = format_response_with_ai(context, "Politely say you only handle route tasks (gas, food, eta).")
    return {"response_type": "other", "text_to_speak": text, "data": {}}


def handle_get_eta(location: Optional[str]) -> dict:
    state = ROUTE_STATE['current_trip_state']
    if not location: location = state['destination']
    for stop in state['stops_on_route']:
        if location.lower() in stop['name'].lower():
            return {"response_type": "eta", "text_to_speak": f"ETA to {stop['name']} is {stop['eta_time']}.", "data": {"stop": stop}}
    for am in ROUTE_STATE['potential_amenities']:
        if location.lower() in am['name'].lower():
             return {"response_type": "eta", "text_to_speak": f"{am['name']} is a {am['detour_minutes']} min detour.", "data": {"stop": am}}
    return handle_other(f"Can't find ETA for {location}")


def handle_get_next_stop() -> dict:
    stop = ROUTE_STATE["current_trip_state"]["stops_on_route"][0]
    return {"response_type": "eta", "text_to_speak": f"Next stop is {stop['name']} at {stop['eta_time']}.", "data": {"stop": stop}}


def handle_get_direction(repeat: bool = False) -> dict:
    state = ROUTE_STATE["current_trip_state"]
    idx = state["next_turn_index"]
    if repeat and idx > 0: idx -= 1
    turn = state["turn_by_turn_directions"][idx]
    text = f"In {turn['distance']}, {turn['instruction']}" if turn['distance'] else turn['instruction']
    if not repeat and (idx + 1) < len(state["turn_by_turn_directions"]):
        state["next_turn_index"] = idx + 1
    return {"response_type": "show_turn_image", "text_to_speak": text, "data": {"image_asset": turn["image_asset"]}}


def handle_get_traffic() -> dict:
    rep = ROUTE_STATE["current_trip_state"]["live_data"]["traffic_ahead"]
    return {"response_type": "info", "text_to_speak": rep, "data": {"report": rep}}


def handle_get_speed_limit() -> dict:
    sl = ROUTE_STATE["current_trip_state"]["live_data"]["current_speed_limit"]
    return {"response_type": "info", "text_to_speak": f"Speed limit is {sl}.", "data": {"speed_limit": sl}}


if __name__ == '__main__':
    print("Starting Dash AI Server (v23 - Gas Fix) at http://127.0.0.1:5001")
    socketio.run(app, port=5001, allow_unsafe_werkzeug=True)
