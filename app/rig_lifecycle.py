"""Bounded controller-manager startup using issue #29's ROS service contract."""
from navigation.types import SafeStop


def controller_plan(state, type_name, reload=False):
    if state not in (None, "active", "inactive", "unconfigured"):
        raise SafeStop(f"unsupported controller state: {state}")
    if state is None:
        if not type_name:
            raise SafeStop("missing controller has no configured type")
        return ["load", "configure", "activate"]
    if state == "unconfigured":
        return ["configure", "activate"]
    if reload:
        prefix = ["deactivate"] if state == "active" else []
        return prefix + (["unload", "load", "configure"] if type_name else []) + ["activate"]
    return [] if state == "active" else ["activate"]


def select_controller(states, configured_types, candidates, explicit=None):
    if explicit:
        if explicit not in candidates:
            raise SafeStop("controller override must be a supported arm controller")
        others = [n for n in candidates if n != explicit and states.get(n) == "active"]
        if others:
            raise SafeStop(f"another arm controller is active: {others}")
        return explicit
    active = [n for n in candidates if states.get(n) == "active"]
    present = [n for n in candidates if n in states]
    choices = active or present or [n for n in candidates if configured_types.get(n)]
    if len(choices) != 1:
        raise SafeStop(f"cannot uniquely select arm controller: {choices}")
    return choices[0]


class RigLifecycle:
    def __init__(self, io):
        import controller_manager_msgs.srv as cms
        self.io, self.cms = io, cms

    def controllers(self, manager):
        res = self.io.call(self.cms.ListControllers, manager + "/list_controllers",
                           self.cms.ListControllers.Request())
        return {c.name: c.state for c in res.controller}, res.controller

    def parameter(self, manager, name):
        from rcl_interfaces.srv import GetParameters
        res = self.io.call(GetParameters, manager + "/get_parameters",
                           GetParameters.Request(names=[name]), optional=True)
        if res and res.values and res.values[0].type == 4:
            return res.values[0].string_value or None
        return None

    def switch(self, manager, name, activate):
        srv = self.cms.SwitchController
        request = srv.Request()
        request.activate_controllers = [name] if activate else []
        request.deactivate_controllers = [] if activate else [name]
        request.strictness = srv.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout.sec = 8
        res = self.io.call(srv, manager + "/switch_controller", request, timeout=10)
        if not res or not res.ok:
            raise SafeStop(f"controller switch rejected: {manager}/{name}")

    def step(self, manager, name, step):
        if step in ("activate", "deactivate"):
            self.switch(manager, name, step == "activate")
        else:
            srv = {"load": self.cms.LoadController, "unload": self.cms.UnloadController,
                   "configure": self.cms.ConfigureController}[step]
            res = self.io.call(srv, manager + f"/{step}_controller", srv.Request(name=name), timeout=10)
            if not res or not res.ok:
                raise SafeStop(f"controller {step} rejected: {manager}/{name}")
        expected = {"activate": "active", "deactivate": "inactive", "unload": None,
                    "load": "unconfigured", "configure": "inactive"}[step]
        states, _ = self.controllers(manager)
        if states.get(name) != expected:
            raise SafeStop(f"controller {step} not confirmed: {manager}/{name}: {states}")

    def arms(self):
        cfg = self.io.cfg["rig"]
        # No arm target exists before this one-time startup sequence. Each side
        # begins its uninterrupted stream only after any deactivate/reload steps.
        for side in ("right", "left"):
            manager = cfg["controller_managers"][side]
            states, _ = self.controllers(manager)
            types = {n: self.parameter(manager, n + ".type") for n in cfg["controller_names"]}
            name = select_controller(states, types, cfg["controller_names"],
                                     cfg["arm_controller_overrides"].get(side))
            plan = controller_plan(states.get(name), types[name],
                                   cfg["reload_impedance"] and name == "joint_impedance_controller")
            self.io.log("controller_startup", side=side, manager=manager, controller=name,
                        state=states.get(name), type=types[name], plan=plan)
            if not plan:
                self.io.hold_current(side)
            for step in plan:
                if step == "activate":
                    self.io.hold_current(side)
                self.step(manager, name, step)

    def base(self):
        from lifecycle_msgs.msg import State
        cfg = self.io.cfg["rig"]
        manager = cfg["controller_managers"]["base"]
        srv = self.cms.ListHardwareComponents
        res = self.io.call(srv, manager + "/list_hardware_components", srv.Request(), optional=True)
        if res is None:
            self.io.log("base_hardware_service_absent", manager=manager)
            listed = self.io.call(self.cms.ListControllers, manager + "/list_controllers",
                                  self.cms.ListControllers.Request(), optional=True)
            if listed is not None:
                states = {c.name: c.state for c in listed.controller}
                for step in controller_plan(states.get(cfg["base_controller"]), None):
                    self.step(manager, cfg["base_controller"], step)
            return  # HKUST permits rigs without this optional manager service.
        states, controllers = self.controllers(manager)
        base_name = cfg["base_controller"]
        if base_name not in states:
            raise SafeStop(f"base controller missing from {manager}: {base_name}")
        components = {c.name: c for c in res.component}
        names = list(cfg["base_hardware_names"])
        if not names:
            base = next(c for c in controllers if c.name == base_name)
            required = set(getattr(base, "required_command_interfaces", [])) | set(base.claimed_interfaces)
            names = [n for n, c in components.items()
                     if required.intersection(i.name for i in c.command_interfaces)]
            if not names and len(components) == 1:
                names = list(components)
        if not names or any(n not in components for n in names):
            raise SafeStop("cannot identify base hardware; set rig.base_hardware_names")
        for name in names:
            state = components[name].state.label
            if state == "active":
                continue
            if state != "inactive":
                raise SafeStop(f"base hardware {name} is {state}; refusing unknown recovery")
            target = State(id=State.PRIMARY_STATE_ACTIVE, label="active")
            activate = self.cms.SetHardwareComponentState
            result = self.io.call(activate, manager + "/set_hardware_component_state",
                                  activate.Request(name=name, target_state=target), timeout=15)
            if not result or not result.ok:
                raise SafeStop(f"base hardware activation rejected: {name}")
        confirmed = self.io.call(srv, manager + "/list_hardware_components", srv.Request())
        after = {c.name: c.state.label for c in confirmed.component}
        if any(after.get(n) != "active" for n in names):
            raise SafeStop(f"base hardware activation not confirmed: {after}")
        for step in controller_plan(states[base_name], None):
            self.step(manager, base_name, step)
        self.io.log("base_hardware_ready", manager=manager, names=names)
