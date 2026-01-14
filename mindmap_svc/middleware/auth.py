import base64
import json
import os
import requests

from typing import Dict
import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


def api_key_decoder(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    api_key解析器
    """
    api_key = credentials.credentials
    return api_key


def token_decoder(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict | None:
    """
    token解析器
    """
    token = credentials.credentials
    try:
        header, payload, signature = token.split('.')
        payload = json.loads(base64.urlsafe_b64decode(payload + '=='))
        payload["token"] = token
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token已过期"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token无效"
        )

    # 业务层依赖字段检查
    if "user_name" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token缺失user_name"
        )

    return payload


def _decode_token(token) -> Dict | None:
    if token is None:
        return None
    _k_sys = os.environ['KB_API_SERVICE_AUTHENTICATION']
    token_str = token[7:]
    # account_service="http://{}/users/userinfo?token={}".format(_account_mgr,token)
    sys_service = "http://{}/token/userinfo?token={}".format(_k_sys, token_str)
    headers = {
        "Authorization": token,
        # "user_name": f"{user_name}",
        "access-key": "1effa9c3b68447c7b8dcac84ccaebd32"
    }
    return {
        "Authorization": token}
        # "user_name": f"{user_name}"}
    try:
        rp = requests.get(url=sys_service, headers=headers)
        if rp.ok:
            rp_json = rp.json()
            if rp_json['code'] == 200:
                return rp_json['data']
            else:
                print("Parsing token exception occurred: {}".format(json.dumps(rp_json, ensure_ascii=False)))
        else:
            print("Invoking k-sys exception occurred:{}".format(rp.status_code))
    except Exception as e:
        print("Parsing token exception occurred: {}".format(e))

    return None
