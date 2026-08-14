from client import Client


def request(method: str) -> str:
    with Client() as client:
        return client.request(method)


def get() -> str:
    return request("GET")
