import react_agent
import react_agent.agent


def test_package_importable():
    assert react_agent is not None
    assert react_agent.agent is not None
