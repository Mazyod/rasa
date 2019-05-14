def test_shell_help(run):
    help = run("shell", "--help")

    help_text = """usage: rasa shell [-h] [-v] [-vv] [--quiet] [-m MODEL] [--log-file LOG_FILE]
                  [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
                  [--cors [CORS [CORS ...]]] [--enable-api]
                  [--remote-storage REMOTE_STORAGE]
                  [--credentials CREDENTIALS] [--connector CONNECTOR]
                  [--jwt-secret JWT_SECRET] [--jwt-method JWT_METHOD]
                  {nlu} ... [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_shell_nlu_help(run):
    help = run("shell", "nlu", "--help")

    help_text = """usage: rasa shell nlu [-h] [-v] [-vv] [--quiet] [-m MODEL]
                      [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line
