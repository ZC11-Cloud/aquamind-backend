import os
import random
import time

from locust import HttpUser, between, task


class RagUser(HttpUser):
    wait_time = between(1, 3)
    token: str | None = None
    conversation_id: int | None = None
    ready: bool = False
    username: str = ""
    password: str = "Test123456"
    chat_mode: str = os.getenv("AQUAMIND_PERF_CHAT_MODE", "mixed").strip().lower()

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _is_rag_enabled(self) -> bool:
        if self.chat_mode not in {"mixed", "rag_only", "normal_only"}:
            self.chat_mode = "mixed"
        return self.chat_mode in {"mixed", "rag_only"}

    def _is_normal_enabled(self) -> bool:
        if self.chat_mode not in {"mixed", "rag_only", "normal_only"}:
            self.chat_mode = "mixed"
        return self.chat_mode in {"mixed", "normal_only"}

    def _try_register(self) -> bool:
        register_payload = {
            "username": self.username,
            "password": self.password,
            "real_name": "Perf User",
            "phone": "13800000000",
            "email": f"{self.username}@example.com",
        }
        resp = self.client.post(
            "/user/register",
            json=register_payload,
            name="/user/register",
        )
        # 200: 创建成功；400: 用户已存在（视为可继续）
        return resp.status_code in (200, 400)

    def _try_login(self) -> bool:
        resp = self.client.post(
            "/user/login",
            data={"username": self.username, "password": self.password},
            name="/user/login",
        )
        if resp.status_code != 200:
            return False
        token = resp.json().get("access_token")
        if not token:
            return False
        self.token = token
        return True

    def _try_create_conversation(self) -> bool:
        resp = self.client.post(
            "/qa/conversations",
            json={"title": "性能测试会话"},
            headers=self._auth_headers(),
            name="/qa/conversations:create",
        )
        if resp.status_code != 201:
            return False
        self.conversation_id = resp.json().get("id")
        return bool(self.conversation_id)

    def on_start(self) -> None:
        # 使用时间戳+随机数组合，避免并发下用户名冲突导致初始化抖动
        self.username = f"perf_user_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        max_attempts = 3
        for _ in range(max_attempts):
            if not self._try_register():
                time.sleep(0.2)
                continue
            if not self._try_login():
                time.sleep(0.2)
                continue
            if not self._try_create_conversation():
                time.sleep(0.2)
                continue
            self.ready = True
            break

        if not self.ready:
            # 初始化失败用户不再发业务请求，避免制造无效401噪声
            self.token = None
            self.conversation_id = None

    @task(4)
    def rag_query(self) -> None:
        if not self.ready or not self.conversation_id:
            return
        if not self._is_rag_enabled():
            return
        payload = {
            "content": random.choice(
                [
                    "什么是蜻蜓稚虫",
                    "轮虫属于哪一类生物",
                    "甲壳类生物有哪些典型特征",
                    "为什么蜉蝣稚虫可作为水质指示生物",
                    "腹足类软体动物的代表特点是什么",
                ]
            ),
            "use_rag": True,
        }
        with self.client.post(
            f"/qa/conversations/{self.conversation_id}/messages",
            json=payload,
            headers=self._auth_headers(),
            name="/qa/conversations/:id/messages:rag",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"status={resp.status_code}")

    @task(2)
    def general_query(self) -> None:
        if not self.ready or not self.conversation_id:
            return
        if not self._is_normal_enabled():
            return
        payload = {
            "content": "请简述RAG系统为什么需要引用来源",
            "use_rag": False,
        }
        with self.client.post(
            f"/qa/conversations/{self.conversation_id}/messages",
            json=payload,
            headers=self._auth_headers(),
            name="/qa/conversations/:id/messages:normal",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def list_messages(self) -> None:
        if not self.ready or not self.conversation_id:
            return
        with self.client.get(
            f"/qa/conversations/{self.conversation_id}/messages?skip=0&limit=20",
            headers=self._auth_headers(),
            name="/qa/conversations/:id/messages:list",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def search_knowledge(self) -> None:
        if not self.ready:
            return
        with self.client.get(
            "/knowledge/search?q=轮虫&top_k=5",
            headers=self._auth_headers(),
            name="/knowledge/search",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
