# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses


@dataclasses.dataclass
class Conversation:
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: list[str] = (("USER", "ASSISTANT"),)
    messages: list[list[str]] = ()
    sep_style: str = ""
    sep: str = "\n"
    sep2: str = None
    stop_token_ids: list[int] = None

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.name == "llama-ptuning":
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += role + " " + message + seps[i % 2]
                else:
                    ret += role
            return ret
        
    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_token_ids=self.stop_token_ids,
        )


conv_templates: dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()

register_conv_template(
    Conversation(
        name="llama-ptuning",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep=" ",
        sep2=" </s><s>",
    )
)



if __name__ == "__main__":
    print("llama-ptuning template:")
    conv = get_conv_template("llama-ptuning")
    conv.set_system_message("You are a helpful, respectful and honest assistant.")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())