# feedback.py
import json
feedback_store = {}  # {rec_item: {'boost': 0.0, 'count': 0}}

def update_feedback(rec_item_id, thumbs_up=True):
    item = str(rec_item_id)
    if item not in feedback_store:
        feedback_store[item] = {'boost': 0.0, 'count': 0}
    delta = 0.2 if thumbs_up else -0.3
    feedback_store[item]['boost'] += delta / (feedback_store[item]['count'] + 1)
    feedback_store[item]['count'] += 1
    with open('data/feedback.json', 'w') as f:
        json.dump(feedback_store, f)

def apply_feedback_adjustment(sim_scores, rec_items):
    for i, item in enumerate(rec_items):
        boost = feedback_store.get(str(item), {}).get('boost', 0.0)
        sim_scores[i] += boost  # Re-rank with boost
    return sim_scores  # Integrate into recommend()
