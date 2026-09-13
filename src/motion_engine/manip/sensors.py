"""Sensor readout for the manipulation cells: poses, contact forces and the derived quantities the
cells report.
"""
from __future__ import annotations

from newton import Contacts
from newton.sensors import SensorContact


class ContactSensor:
    """SensorContact + Contacts for contact-force readout across W worlds.

    The default patterns are the contact-pick chain's: the force on the end-effector sphere (ee_tip)
    from the cube (cube_shape). force() returns the (W, 3) matrix, the same
    reshape som legacy (`sensor.force_matrix.numpy().reshape(W, 3)`).
    """

    def __init__(self, model, solver, *, W: int,
                 sensing_obj_shapes: str = "*ee_tip*",
                 counterpart_shapes: str = "*cube_shape*"):
        self.solver = solver
        self.W = W
        self.sensor = SensorContact(model, sensing_obj_shapes=sensing_obj_shapes,
                                    counterpart_shapes=counterpart_shapes)
        self.contacts = Contacts(solver.get_max_contact_count(), 0,
                                 requested_attributes=model.get_requested_contact_attributes())

    def force(self, s0):
        """Force on the sensing shape from its counterpart, (W, 3)."""
        self.solver.update_contacts(self.contacts, s0)
        self.sensor.update(s0, self.contacts)
        return self.sensor.force_matrix.numpy().reshape(self.W, 3)
