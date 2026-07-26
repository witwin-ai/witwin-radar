"""Name what each composed row IS: target echo, clutter, leakage, multibounce.

A radar frame is never one target. It is a target sitting inside an environment
that returns energy of its own, and every useful thing a radar engineer wants
to do with that - export the clutter alone, subtract it, measure the target
against it, recombine the two coherently - needs the rows to carry a NAME.
Channel publishes the ingredients of that name on every frozen leg row
(``component_id``, ``depth``, ``material_sequence``, the endpoint IDs) and the
composed topology publishes the map back to them (``inbound_row`` and
``outbound_row``). What was missing was the name itself.

This module is that name, and it is deliberately a SIDECAR.
:class:`~witwin.radar.paths.contracts.RadarPathTopology` gains no column, so
the frozen dataclass, its identity tests and every existing consumer are
untouched, and the acceptance criterion "processing does not change propagation
row identity" becomes directly assertable: every component export shares the
SAME topology object, checked with ``is``.

Three properties, and each is a decision:

**It is built once, at freeze time.** The classification is a function of the
frozen topology and the frozen legs alone - not of any frame's payload - so it
is host work performed exactly once per topology epoch and costs a frame
nothing. It reads the frozen identity columns to the host through ``tolist``,
which is the same sanctioned freeze-time observation
:mod:`witwin.radar.paths._identity` already makes, after the consumer has
already synchronized.

**It is a declaration, not an inference.** Which sites are targets and which
geometry is clutter is a statement about the SCENE that no tensor can answer:
the same wall is clutter to an automotive radar and the target of a
through-wall imager. :class:`ComponentDeclaration` is where the caller says so,
and it is recorded on the index so a report can quote it.

**Every row belongs to exactly one class.** The four predicates are evaluated
independently and the build refuses a row that matches none (an undeclared
site, which would otherwise be silently dropped from every export and from the
coherent recombination law) and a row that matches two (a declaration that
contradicts itself). Silence in either direction would make ``sum_j
cube(component_j) == cube(all)`` quietly false.

A structure is named by its COMPILED MATERIAL SLOT rather than by its authored
``structure_id``, because the material slot is what the frozen row actually
carries: Channel publishes ``material_sequence`` and no structure column. The
consequence is stated rather than hidden - two structures sharing one material
are indistinguishable here, and a scene that needs them apart must give them
different materials or declare its clutter by site instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import RadarPathTopology
from .direct import NO_SITE


#: A row whose round trip scatters off a declared target site and touches no
#: declared clutter geometry on the way.
TARGET = "target"

#: A row that scatters off declared clutter geometry, or off a site the caller
#: declared to be clutter. It is not a lesser target echo: it is the return the
#: environment makes on its own, and it is coherent with the target return
#: because it lives in the same pair segment of the same waveform kernel.
ENVIRONMENT_CLUTTER = "environment_clutter"

#: The direct transmitter-to-receiver route: no scatter site and no declared
#: clutter interaction. A site-less row that DOES touch declared clutter
#: geometry - transmitter to wall to receiver - is environment clutter, not
#: leakage, because it is the environment's return and not the antenna
#: coupling term.
DIRECT_LEAKAGE = "direct_leakage"

#: A round trip that interacted more than the declared depth on either leg.
#: Kept separate from clutter because a multibounce return through a target is
#: neither a clean target echo nor an environment-only one.
MULTI_INTERACTION = "multi_interaction"

#: The declared taxonomy, in the order the index publishes it. Every class is
#: always published, including an empty one: a caller that sums the per-class
#: cubes must be able to iterate a fixed list rather than discover which classes
#: this particular frame happened to produce.
COMPONENT_NAMES: tuple[str, ...] = (
    TARGET,
    ENVIRONMENT_CLUTTER,
    DIRECT_LEAKAGE,
    MULTI_INTERACTION,
)

#: The value both interaction sequences carry where a row interacted with
#: nothing. Channel publishes it on every line-of-sight row.
NO_INTERACTION = -1


def _int_set(values: object, name: str) -> frozenset[int]:
    if values is None:
        return frozenset()
    if isinstance(values, torch.Tensor):
        raise TypeError(
            f"{name} is a host declaration about the scene, not a device "
            "tensor; pass a set of stable IDs"
        )
    if isinstance(values, (int, str)):
        raise TypeError(f"{name} must be an iterable of ints, got {type(values).__name__}")
    return frozenset(int(value) for value in values)


@dataclass(frozen=True, slots=True, eq=False)
class ComponentDeclaration:
    """What the caller says this scene contains.

    ``target_site_ids`` and ``clutter_site_ids`` are stable world IDs of scatter
    sites. ``clutter_material_slots`` are COMPILED material slots, which is what
    a frozen leg row carries in ``material_sequence``; a round trip that
    interacts with one of them is environment clutter no matter which site it
    also reached, because an echo that bounced off the wall on its way to the
    target is not a clean target echo.

    ``multi_interaction_depth`` is the deepest leg still treated as a simple
    return. The default ``1`` keeps single-bounce reflections in the clutter or
    target classes and sends anything deeper to
    :data:`MULTI_INTERACTION`; it is the Phase-5 ``hybrid`` distinction
    (target echo / environment clutter / multi-interaction echo) expressed as a
    declaration over ONE topology rather than as a third join mode.

    A site declared both target and clutter is refused here rather than
    resolved: the two exports would overlap and the coherent recombination law
    would double-count it.
    """

    target_site_ids: frozenset[int] = frozenset()
    clutter_site_ids: frozenset[int] = frozenset()
    clutter_material_slots: frozenset[int] = frozenset()
    multi_interaction_depth: int = 1

    def __post_init__(self) -> None:
        for name in (
            "target_site_ids",
            "clutter_site_ids",
            "clutter_material_slots",
        ):
            object.__setattr__(self, name, _int_set(getattr(self, name), name))
        if type(self.multi_interaction_depth) is not int or (
            self.multi_interaction_depth < 0
        ):
            raise ValueError(
                "multi_interaction_depth must be a non-negative int, got "
                f"{self.multi_interaction_depth!r}"
            )
        overlap = self.target_site_ids & self.clutter_site_ids
        if overlap:
            raise ValueError(
                f"sites {sorted(overlap)} are declared both target and clutter; "
                "the two component exports would overlap and their coherent sum "
                "would count those rows twice"
            )

    def classify(
        self, *, site_id: int, depth: int, material_slots: frozenset[int]
    ) -> tuple[str, ...]:
        """Every class this row belongs to, evaluated independently.

        The four predicates are written separately rather than as an if/elif
        ladder on purpose. A ladder makes "exactly one class" true by
        construction and therefore unassertable; evaluating all four lets the
        index refuse a declaration that produces none or two.

        ``direct_leakage`` is the route with no scatter site AND no declared
        clutter interaction, which is a narrower predicate than "no site". The
        difference is a real scene: the transmitter-to-wall-to-receiver path
        also has no site, and calling it leakage would file the strongest
        environment return in the frame under the name of the antenna coupling
        term. A site-less row that touches declared clutter geometry is
        environment clutter, which is what it is.
        """

        direct = site_id == NO_SITE
        deep = depth > self.multi_interaction_depth
        clutter = bool(material_slots & self.clutter_material_slots) or (
            site_id in self.clutter_site_ids
        )
        matched: list[str] = []
        if not direct and not deep and not clutter and (
            site_id in self.target_site_ids
        ):
            matched.append(TARGET)
        if not deep and clutter:
            matched.append(ENVIRONMENT_CLUTTER)
        if direct and not deep and not clutter:
            matched.append(DIRECT_LEAKAGE)
        if deep:
            matched.append(MULTI_INTERACTION)
        return tuple(matched)


def _leg_facts(leg, name: str) -> tuple[list[int], list[frozenset[int]]]:
    """One frozen leg's depth and interacted material slots, read once."""

    depth = [int(value) for value in leg.depth.tolist()]
    materials = [
        frozenset(int(value) for value in row if int(value) != NO_INTERACTION)
        for row in leg.material_sequence.tolist()
    ]
    if len(depth) != len(materials):
        raise ValueError(
            f"{name} leg publishes {len(depth)} depths and {len(materials)} "
            "material sequences"
        )
    return depth, materials


@dataclass(frozen=True, slots=True, eq=False)
class RadarComponentIndex:
    """Which component class owns each composed row.

    ``topology`` is held by REFERENCE and is the object every export must
    share. ``class_id`` is ``[path_count]`` int32 on the batch device, indexing
    :attr:`names`. ``counts`` is the same partition counted on the host at build
    time, so a test or a report can say "this class has four rows" without
    reading the device.
    """

    topology: RadarPathTopology
    class_id: torch.Tensor
    names: tuple[str, ...]
    counts: tuple[int, ...]
    declaration: ComponentDeclaration

    @property
    def row_count(self) -> int:
        return int(self.class_id.shape[0])

    def index_of(self, name: str) -> int:
        if name not in self.names:
            raise KeyError(
                f"{name!r} is not a declared component; this index publishes "
                f"{list(self.names)}"
            )
        return self.names.index(name)

    def count(self, name: str) -> int:
        """How many rows this class owns. A host int, decided at build time."""

        return self.counts[self.index_of(name)]

    def mask(self, name: str) -> torch.Tensor:
        """``[path_count]`` bool selecting this class's rows.

        Derived from ``class_id`` rather than stored, so no caller can hold a
        reference to a mask and mutate the index behind another caller's back.
        """

        return self.class_id == self.index_of(name)

    @classmethod
    def from_two_way(
        cls,
        composer,
        inbound,
        outbound,
        declaration: ComponentDeclaration,
    ) -> "RadarComponentIndex":
        """Classify a two-way join's rows from its two frozen leg topologies.

        ``composer`` is a :class:`~witwin.radar.paths.two_way.TwoWayComposer`
        and ``inbound`` / ``outbound`` are the frozen leg handles it was frozen
        against; all three are duck typed so this module adds no import edge to
        the Channel adapter.
        """

        return cls._build(composer.topology, inbound, outbound, declaration)

    @classmethod
    def from_direct(
        cls, composer, leg, declaration: ComponentDeclaration
    ) -> "RadarComponentIndex":
        """Classify a direct composer's rows. There is no second leg."""

        return cls._build(composer.topology, leg, None, declaration)

    @classmethod
    def _build(
        cls,
        topology: RadarPathTopology,
        inbound,
        outbound,
        declaration: ComponentDeclaration,
    ) -> "RadarComponentIndex":
        if not isinstance(declaration, ComponentDeclaration):
            raise TypeError(
                "declaration must be a ComponentDeclaration, got "
                f"{type(declaration).__name__}"
            )
        site = [int(value) for value in topology.site_id.tolist()]
        inbound_row = [int(value) for value in topology.inbound_row.tolist()]
        outbound_row = [int(value) for value in topology.outbound_row.tolist()]
        in_depth, in_materials = _leg_facts(inbound, "inbound")
        if outbound is None:
            out_depth, out_materials = [], []
        else:
            out_depth, out_materials = _leg_facts(outbound, "outbound")

        classes: list[int] = []
        for row, site_id in enumerate(site):
            depth = in_depth[inbound_row[row]]
            materials = in_materials[inbound_row[row]]
            if outbound is not None and outbound_row[row] >= 0:
                depth = max(depth, out_depth[outbound_row[row]])
                materials = materials | out_materials[outbound_row[row]]
            matched = declaration.classify(
                site_id=site_id, depth=depth, material_slots=materials
            )
            if len(matched) != 1:
                raise ValueError(
                    f"composed row {row} (site {site_id}, depth {depth}, "
                    f"material slots {sorted(materials)}) matches "
                    f"{list(matched)}; every row must belong to exactly one "
                    "component class, so a row matching none is an undeclared "
                    "site and a row matching two is a declaration that "
                    "contradicts itself"
                )
            classes.append(COMPONENT_NAMES.index(matched[0]))

        counts = tuple(
            sum(1 for value in classes if value == index)
            for index in range(len(COMPONENT_NAMES))
        )
        return cls(
            topology=topology,
            class_id=torch.tensor(
                classes, dtype=torch.int32, device=topology.site_id.device
            ),
            names=COMPONENT_NAMES,
            counts=counts,
            declaration=declaration,
        )


__all__ = [
    "COMPONENT_NAMES",
    "DIRECT_LEAKAGE",
    "ENVIRONMENT_CLUTTER",
    "MULTI_INTERACTION",
    "NO_INTERACTION",
    "TARGET",
    "ComponentDeclaration",
    "RadarComponentIndex",
]
