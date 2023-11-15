import ssl
from functools import cached_property

import aiohttp
import logging
import os
from aiohttp.client_exceptions import ContentTypeError
from typing import Any, Optional, Text, Dict

from rasa.shared.exceptions import FileNotFoundException
import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT


logger = logging.getLogger(__name__)


def read_endpoint_config(
    filename: Text, endpoint_type: Text
) -> Optional["EndpointConfig"]:
    """Read an endpoint configuration file from disk and extract one config."""  # noqa: E501
    if not filename:
        return None

    try:
        content = rasa.shared.utils.io.read_config_file(filename)

        if content.get(endpoint_type) is None:
            return None

        return EndpointConfig.from_dict(content[endpoint_type])
    except FileNotFoundError:
        logger.error(
            "Failed to read endpoint configuration "
            "from {}. No such file.".format(os.path.abspath(filename))
        )
        return None


def concat_url(base: Text, subpath: Optional[Text]) -> Text:
    """Append a subpath to a base url.

    Strips leading slashes from the subpath if necessary. This behaves
    differently than `urlparse.urljoin` and will not treat the subpath
    as a base url if it starts with `/` but will always append it to the
    `base`.

    Args:
        base: Base URL.
        subpath: Optional path to append to the base URL.

    Returns:
        Concatenated URL with base and subpath.
    """
    if not subpath:
        if base.endswith("/"):
            logger.debug(
                f"The URL '{base}' has a trailing slash. Please make sure the "
                f"target server supports trailing slashes for this endpoint."
            )
        return base

    url = base
    if not base.endswith("/"):
        url += "/"
    if subpath.startswith("/"):
        subpath = subpath[1:]
    return url + subpath


class EndpointConfig:
    """Configuration for an external HTTP endpoint."""

    def __init__(
        self,
        url: Optional[Text] = None,
        params: Optional[Dict[Text, Any]] = None,
        headers: Optional[Dict[Text, Any]] = None,
        basic_auth: Optional[Dict[Text, Text]] = None,
        token: Optional[Text] = None,
        token_name: Text = "token",
        cafile: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        """Creates an `EndpointConfig` instance."""
        self.url = url
        self.params = params or {}
        self.headers = headers or {}
        self.basic_auth = basic_auth or {}
        self.token = token
        self.token_name = token_name
        self.type = kwargs.pop("store_type", kwargs.pop("type", None))
        self.cafile = cafile
        self.kwargs = kwargs

    @cached_property
    def session(self) -> aiohttp.ClientSession:
        """Creates and returns a configured aiohttp client session."""
        # create authentication parameters
        if self.basic_auth:
            auth = aiohttp.BasicAuth(
                self.basic_auth["username"], self.basic_auth["password"]
            )
        else:
            auth = None

        return aiohttp.ClientSession(
            headers=self.headers,
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT),
        )

    def combine_parameters(
        self, kwargs: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        # construct GET parameters
        params = self.params.copy()

        # set the authentication token if present
        if self.token:
            params[self.token_name] = self.token

        if kwargs and "params" in kwargs:
            params.update(kwargs["params"])
            del kwargs["params"]
        return params

    async def request(
        self,
        method: Text = "post",
        subpath: Optional[Text] = None,
        content_type: Optional[Text] = "application/json",
        compress: bool = False,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Send a HTTP request to the endpoint. Return json response, if available.

        All additional arguments will get passed through
        to aiohttp's `session.request`.
        """
        # create the appropriate headers
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            del kwargs["headers"]

        if self.headers:
            headers.update(self.headers)

        url = concat_url(self.url, subpath)

        sslcontext = None
        if self.cafile:
            try:
                sslcontext = ssl.create_default_context(cafile=self.cafile)
            except FileNotFoundError as e:
                raise FileNotFoundException(
                    f"Failed to find certificate file, "
                    f"'{os.path.abspath(self.cafile)}' does not exist."
                ) from e

        async with self.session.request(
            method,
            url,
            headers=headers,
            params=self.combine_parameters(kwargs),
            compress=compress,
            ssl=sslcontext,
            **kwargs,
        ) as response:
            if response.status >= 400:
                raise ClientResponseError(
                    response.status, response.reason, await response.content.read()
                )
            try:
                return await response.json()
            except ContentTypeError:
                return None

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "EndpointConfig":
        return EndpointConfig(**data)

    def copy(self) -> "EndpointConfig":
        return EndpointConfig(
            self.url,
            self.params,
            self.headers,
            self.basic_auth,
            self.token,
            self.token_name,
            **self.kwargs,
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, type(other)):
            return (
                other.url == self.url
                and other.params == self.params
                and other.headers == self.headers
                and other.basic_auth == self.basic_auth
                and other.token == self.token
                and other.token_name == self.token_name
            )
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class ClientResponseError(aiohttp.ClientError):
    def __init__(self, status: int, message: Text, text: Text) -> None:
        self.status = status
        self.message = message
        self.text = text
        super().__init__(f"{status}, {message}, body='{text}'")


def float_arg(
    request: "Request", key: Text, default: Optional[float] = None
) -> Optional[float]:
    """Returns a passed argument cast as a float or None.

    Checks the `key` parameter of the request if it contains a valid
    float value. If not, `default` is returned.

    Args:
        request: Sanic request.
        key: Name of argument.
        default: Default value for `key` argument.

    Returns:
        A float value if `key` is a valid float, `default` otherwise.
    """
    arg = request.args.get(key, default)

    if arg is default:
        return arg

    try:
        return float(str(arg))
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert '{arg}' to float.")
        return default


def int_arg(
    request: "Request", key: Text, default: Optional[int] = None
) -> Optional[int]:
    """Returns a passed argument cast as an int or None.

    Checks the `key` parameter of the request if it contains a valid
    int value. If not, `default` is returned.

    Args:
        request: Sanic request.
        key: Name of argument.
        default: Default value for `key` argument.

    Returns:
        An int value if `key` is a valid integer, `default` otherwise.
    """
    arg = request.args.get(key, default)

    if arg is default:
        return arg

    try:
        return int(str(arg))
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert '{arg}' to int.")
        return default
