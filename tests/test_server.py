# -*- coding: utf-8 -*-
import pytest

import rasa
import rasa.constants
from tests.nlu.conftest import NLU_MODEL_NAME
from tests.nlu.utilities import ResponseTest


@pytest.fixture
def rasa_app(rasa_server):
    return rasa_server.test_client


@pytest.fixture
def rasa_secured_app(rasa_server_secured):
    return rasa_server_secured.test_client


def test_root(rasa_app):
    _, response = rasa_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_root_secured(rasa_secured_app):
    _, response = rasa_secured_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_version(rasa_app):
    _, response = rasa_app.get("/version")
    content = response.json
    assert response.status == 200
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


def test_status(rasa_app):
    _, response = rasa_app.get("/status")
    assert response.status == 200


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"text": "hello ńöñàśçií"},
        ),
    ],
)
def test_post_parse(rasa_app, response_test):
    _, response = rasa_app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["text"] == response_test.expected_response["text"]


#
# @pytest.mark.parametrize(
#     "response_test",
#     [
#         ResponseTest(
#             "/parse?q=hello&model=some-model",
#             {
#                 "entities": [],
#                 "model": "fallback",
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello",
#             },
#         ),
#         ResponseTest(
#             "/parse?query=hello&model=some-model",
#             {
#                 "entities": [],
#                 "model": "fallback",
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello",
#             },
#         ),
#         ResponseTest(
#             "/parse?q=hello ńöñàśçií&model=some-model",
#             {
#                 "entities": [],
#                 "model": "fallback",
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello ńöñàśçií",
#             },
#         ),
#         ResponseTest(
#             "/parse?q=&model=abc",
#             {
#                 "entities": [],
#                 "model": "fallback",
#                 "intent": {"confidence": 0.0, "name": None},
#                 "text": "",
#             },
#         ),
#     ],
# )
# def test_get_parse_use_fallback_model(app_without_model, response_test):
#     _, response = app_without_model.get(response_test.endpoint)
#     rjs = response.json
#     assert response.status == 200
#     assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
#     assert rjs["entities"] == response_test.expected_response["entities"]
#     assert rjs["model"] == FALLBACK_MODEL_NAME
#     assert rjs["text"] == response_test.expected_response["text"]
#
#
# @pytest.mark.parametrize(
#     "response_test",
#     [
#         ResponseTest(
#             "/parse",
#             {
#                 "entities": [],
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello",
#             },
#             payload={"q": "hello", "model": "some-model"},
#         ),
#         ResponseTest(
#             "/parse",
#             {
#                 "entities": [],
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello",
#             },
#             payload={"query": "hello", "model": "some-model"},
#         ),
#         ResponseTest(
#             "/parse",
#             {
#                 "entities": [],
#                 "intent": {"confidence": 1.0, "name": "greet"},
#                 "text": "hello ńöñàśçií",
#             },
#             payload={"q": "hello ńöñàśçií", "model": "some-model"},
#         ),
#     ],
# )
# def test_post_parse_using_fallback_model(app, response_test):
#     _, response = app.post(response_test.endpoint, json=response_test.payload)
#     rjs = response.json
#     assert response.status == 200
#     assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
#     assert rjs["entities"] == response_test.expected_response["entities"]
#     assert rjs["model"] == FALLBACK_MODEL_NAME
#     assert rjs["text"] == response_test.expected_response["text"]
#     assert rjs["intent"]["name"] == response_test.expected_response["intent"]["name"]
#
#
# @utilities.slowtest
# def test_post_train_success(app_without_model, rasa_default_train_data):
#     request = {
#         "language": "en",
#         "pipeline": "pretrained_embeddings_spacy",
#         "data": rasa_default_train_data,
#     }
#
#     _, response = app_without_model.post("/train", json=request)
#
#     assert response.status == 200
#     assert response.content is not None
#
#
# @utilities.slowtest
# def test_post_train_internal_error(app, rasa_default_train_data):
#     _, response = app.post(
#         "/train", json={"data": "dummy_data_for_triggering_an_error"}
#     )
#     assert response.status == 500, "The training data format is not valid"
#
#
# def test_model_hot_reloading(app, rasa_default_train_data):
#     query = "/parse?q=hello&model=test-model"
#
#     # Model could not be found, fallback model was used instead
#     _, response = app.get(query)
#     assert response.status == 200
#     rjs = response.json
#     assert rjs["model"] == FALLBACK_MODEL_NAME
#
#     # Train a new model - model will be loaded automatically
#     train_u = "/train?model=test-model"
#     request = {
#         "language": "en",
#         "pipeline": "pretrained_embeddings_spacy",
#         "data": rasa_default_train_data,
#     }
#     model_str = yaml.safe_dump(request, default_flow_style=False, allow_unicode=True)
#     _, response = app.post(
#         train_u, headers={"Content-Type": "application/x-yml"}, data=model_str
#     )
#     assert response.status == 200, "Training should end successfully"
#
#     _, response = app.post(
#         train_u, headers={"Content-Type": "application/json"}, data=json.dumps(request)
#     )
#     assert response.status == 200, "Training should end successfully"
#
#     # Model should be there now
#     _, response = app.get(query)
#     assert response.status == 200, "Model should now exist after it got trained"
#     rjs = response.json
#     assert "test-model" in rjs["model"]
#
#
# def test_evaluate_invalid_model_error(app, rasa_default_train_data):
#     _, response = app.post("/evaluate?model=not-existing", json=rasa_default_train_data)
#
#     rjs = response.json
#     assert response.status == 500
#     assert "details" in rjs
#     assert rjs["details"]["error"] == "Model with name 'not-existing' is not loaded."
#
#
# def test_evaluate_unsupported_model_error(app_without_model, rasa_default_train_data):
#     _, response = app_without_model.post("/evaluate", json=rasa_default_train_data)
#
#     rjs = response.json
#     assert response.status == 500
#     assert "details" in rjs
#     assert rjs["details"]["error"] == "No model is loaded. Cannot evaluate."
#
#
# def test_evaluate_internal_error(app, rasa_default_train_data):
#     _, response = app.post(
#         "/evaluate", json={"data": "dummy_data_for_triggering_an_error"}
#     )
#     assert response.status == 500, "The training data format is not valid"
#
#
# def test_evaluate(app, rasa_default_train_data):
#     _, response = app.post(
#         "/evaluate?model={}".format(NLU_MODEL_NAME), json=rasa_default_train_data
#     )
#
#     rjs = response.json
#     assert "intent_evaluation" in rjs
#     assert "entity_evaluation" in rjs
#     assert all(
#         prop in rjs["intent_evaluation"]
#         for prop in ["report", "predictions", "precision", "f1_score", "accuracy"]
#     )
#     assert response.status == 200, "Evaluation should start"
#
#
# def test_unload_model_error(app):
#     request = "/models?model=my_model"
#     _, response = app.delete(request)
#     rjs = response.json
#     assert (
#         response.status == 404
#     ), "Model is not loaded and can therefore not be unloaded."
#     assert rjs["details"]["error"] == "Model with name 'my_model' is not loaded."
#
#
# def test_unload_model(app):
#     unload = "/models?model={}".format(NLU_MODEL_NAME)
#     _, response = app.delete(unload)
#     assert response.status == 204, "No Content"
#
#
# def test_status_after_unloading(app):
#     _, response = app.get("/status")
#     rjs = response.json
#     assert response.status == 200
#     assert rjs["loaded_model"] == NLU_MODEL_NAME
#
#     unload = "/models?model={}".format(NLU_MODEL_NAME)
#     _, response = app.delete(unload)
#     assert response.status == 204, "No Content"
#
#     _, response = app.get("/status")
#     rjs = response.json
#     assert response.status == 200
#     assert rjs["loaded_model"] is None


#
# @freeze_time("2018-01-01")
# def test_requesting_non_existent_tracker(app):
#     _, response = app.get("/conversations/madeupid/tracker")
#     content = response.json
#     assert response.status == 200
#     assert content["paused"] is False
#     assert content["slots"] == {"location": None, "cuisine": None}
#     assert content["sender_id"] == "madeupid"
#     assert content["events"] == [
#         {
#             "event": "action",
#             "name": "action_listen",
#             "policy": None,
#             "confidence": None,
#             "timestamp": 1514764800,
#         }
#     ]
#     assert content["latest_message"] == {"text": None, "intent": {}, "entities": []}
#
#
# def test_respond(app):
#     data = json.dumps({"query": "/greet"})
#     _, response = app.post(
#         "/conversations/myid/respond",
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     content = response.json
#     assert response.status == 200
#     assert content == [{"text": "hey there!", "recipient_id": "myid"}]
#
#
# def test_parse(app):
#     data = json.dumps({"q": """/greet{"name": "Rasa"}"""})
#     _, response = app.post(
#         "/parse", data=data, headers={"Content-Type": "application/json"}
#     )
#     content = response.json
#     assert response.status == 200
#     assert content == {
#         "entities": [{"end": 22, "entity": "name", "start": 6, "value": "Rasa"}],
#         "intent": {"confidence": 1.0, "name": "greet"},
#         "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
#         "text": '/greet{"name": "Rasa"}',
#     }
#
#
# @pytest.mark.parametrize("event", test_events)
# def test_pushing_event(app, event):
#     cid = str(uuid.uuid1())
#     conversation = "/conversations/{}".format(cid)
#     data = json.dumps({"query": "/greet"})
#     _, response = app.post(
#         "{}/respond".format(conversation),
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     assert response.json is not None
#     assert response.status == 200
#
#     data = json.dumps(event.as_dict())
#     _, response = app.post(
#         "{}/tracker/events".format(conversation),
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     assert response.json is not None
#     assert response.status == 200
#
#     _, tracker_response = app.get("/conversations/{}/tracker".format(cid))
#     tracker = tracker_response.json
#     assert tracker is not None
#     assert len(tracker.get("events")) == 6
#
#     evt = tracker.get("events")[5]
#     assert Event.from_parameters(evt) == event
#
#
# def test_put_tracker(app):
#     data = json.dumps([event.as_dict() for event in test_events])
#     _, response = app.put(
#         "/conversations/pushtracker/tracker/events",
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     content = response.json
#     assert response.status == 200
#     assert len(content["events"]) == len(test_events)
#     assert content["sender_id"] == "pushtracker"
#
#     _, tracker_response = app.get("/conversations/pushtracker/tracker")
#     tracker = tracker_response.json
#     assert tracker is not None
#     evts = tracker.get("events")
#     assert events.deserialise_events(evts) == test_events
#
#
# def test_sorted_predict(app):
#     data = json.dumps([event.as_dict() for event in test_events[:3]])
#     _, response = app.put(
#         "/conversations/sortedpredict/tracker/events",
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#
#     assert response.status == 200
#
#     _, response = app.post("/conversations/sortedpredict/predict")
#     scores = response.json["scores"]
#     sorted_scores = sorted(scores, key=lambda k: (-k["score"], k["action"]))
#     assert scores == sorted_scores
#
#
# def test_list_conversations(app):
#     data = json.dumps({"query": "/greet"})
#     _, response = app.post(
#         "/conversations/myid/respond",
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     assert response.json is not None
#     assert response.status == 200
#
#     _, response = app.get("/conversations")
#     content = response.json
#     assert response.status == 200
#
#     assert len(content) > 0
#     assert "myid" in content
#
#
# def test_evaluate(app):
#     with open(DEFAULT_STORIES_FILE, "r") as f:
#         stories = f.read()
#     _, response = app.post("/evaluate", data=stories)
#     assert response.status == 200
#     js = response.json
#     assert set(js.keys()) == {
#         "report",
#         "precision",
#         "f1",
#         "accuracy",
#         "actions",
#         "in_training_data_fraction",
#         "is_end_to_end_evaluation",
#     }
#     assert not js["is_end_to_end_evaluation"]
#     assert set(js["actions"][0].keys()) == {
#         "action",
#         "predicted",
#         "confidence",
#         "policy",
#     }
#
#
# def test_stack_training(
#     app,
#     default_domain_path,
#     default_stories_file,
#     default_stack_config,
#     default_nlu_data,
# ):
#     domain_file = open(default_domain_path)
#     config_file = open(default_stack_config)
#     stories_file = open(default_stories_file)
#     nlu_file = open(default_nlu_data)
#
#     payload = dict(
#         domain=domain_file.read(),
#         config=config_file.read(),
#         stories=stories_file.read(),
#         nlu=nlu_file.read(),
#     )
#
#     domain_file.close()
#     config_file.close()
#     stories_file.close()
#     nlu_file.close()
#
#     _, response = app.post("/jobs", json=payload)
#     assert response.status == 200
#
#     # save model to temporary file
#     tempdir = tempfile.mkdtemp()
#     model_path = os.path.join(tempdir, "model.tar.gz")
#     with open(model_path, "wb") as f:
#         f.write(response.body)
#
#     # unpack model and ensure fingerprint is present
#     model_path = unpack_model(model_path)
#     assert os.path.exists(os.path.join(model_path, "fingerprint.json"))
#
#
# def test_intent_evaluation(app, default_nlu_data, trained_stack_model):
#     with open(default_nlu_data, "r") as f:
#         nlu_data = f.read()
#
#     # add evaluation data to model archive
#     zipped_path = add_evaluation_file_to_model(
#         trained_stack_model, nlu_data, data_format="md"
#     )
#
#     # post zipped stack model with evaluation file
#     with open(zipped_path, "r+b") as f:
#         _, response = app.post("/intentEvaluation", data=f.read())
#
#     assert response.status == 200
#     assert set(response.json.keys()) == {"intent_evaluation", "entity_evaluation"}
#
#
# def test_end_to_end_evaluation(app):
#     with open(END_TO_END_STORY_FILE, "r") as f:
#         stories = f.read()
#     _, response = app.post("/evaluate?e2e=true", data=stories)
#     assert response.status == 200
#     js = response.json
#     assert set(js.keys()) == {
#         "report",
#         "precision",
#         "f1",
#         "accuracy",
#         "actions",
#         "in_training_data_fraction",
#         "is_end_to_end_evaluation",
#     }
#     assert js["is_end_to_end_evaluation"]
#     assert set(js["actions"][0].keys()) == {
#         "action",
#         "predicted",
#         "confidence",
#         "policy",
#     }
#
#
# def test_list_conversations_with_jwt(secured_app):
#     # token generated with secret "core" and algorithm HS256
#     # on https://jwt.io/
#
#     # {"user": {"username": "testadmin", "role": "admin"}}
#     jwt_header = {
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
#         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
#         "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
#         "QRre7IWTuIDrCn5AIw"
#     }
#     _, response = secured_app.get("/conversations", headers=jwt_header)
#     assert response.status == 200
#
#     # {"user": {"username": "testuser", "role": "user"}}
#     jwt_header = {
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
#         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdHVzZXIiLCJyb"
#         "2xlIjoidXNlciJ9fQ.JnMTLYd56qut2w9h7hRQlDm1n3l"
#         "HJHOxxC_w7TtwCrs"
#     }
#     _, response = secured_app.get("/conversations", headers=jwt_header)
#     assert response.status == 403
#
#
# def test_get_tracker_with_jwt(secured_app):
#     # token generated with secret "core" and algorithm HS256
#     # on https://jwt.io/
#
#     # {"user": {"username": "testadmin", "role": "admin"}}
#     jwt_header = {
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
#         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
#         "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
#         "QRre7IWTuIDrCn5AIw"
#     }
#     _, response = secured_app.get(
#         "/conversations/testadmin/tracker", headers=jwt_header
#     )
#     assert response.status == 200
#
#     _, response = secured_app.get("/conversations/testuser/tracker", headers=jwt_header)
#     assert response.status == 200
#
#     # {"user": {"username": "testuser", "role": "user"}}
#     jwt_header = {
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
#         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdHVzZXIiLCJyb"
#         "2xlIjoidXNlciJ9fQ.JnMTLYd56qut2w9h7hRQlDm1n3l"
#         "HJHOxxC_w7TtwCrs"
#     }
#     _, response = secured_app.get(
#         "/conversations/testadmin/tracker", headers=jwt_header
#     )
#     assert response.status == 403
#
#     _, response = secured_app.get("/conversations/testuser/tracker", headers=jwt_header)
#     assert response.status == 200
#
#
# def test_list_conversations_with_token(secured_app):
#     _, response = secured_app.get("/conversations?token=rasa")
#     assert response.status == 200
#
#
# def test_list_conversations_with_wrong_token(secured_app):
#     _, response = secured_app.get("/conversations?token=Rasa")
#     assert response.status == 401
#
#
# def test_list_conversations_without_auth(secured_app):
#     _, response = secured_app.get("/conversations")
#     assert response.status == 401
#
#
# def test_list_conversations_with_wrong_jwt(secured_app):
#     jwt_header = {
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
#         "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
#         "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
#         "Gl8eZFVfKXA6jhncgRn-I"
#     }
#     _, response = secured_app.get("/conversations", headers=jwt_header)
#     assert response.status == 401
#
#
# def test_story_export(app):
#     data = json.dumps({"query": "/greet"})
#     _, response = app.post(
#         "/conversations/mynewid/respond",
#         data=data,
#         headers={"Content-Type": "application/json"},
#     )
#     assert response.status == 200
#     _, response = app.get("/conversations/mynewid/story")
#     assert response.status == 200
#     story_lines = response.text.strip().split("\n")
#     assert story_lines == ["## mynewid", "* greet: /greet", "    - utter_greet"]
