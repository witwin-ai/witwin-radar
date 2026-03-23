import witwin.radar as wr


def test_scene_uses_set_and_add_mutators_without_with_aliases():
    scene = wr.Scene(device="cpu")

    returned = scene.set_sensor(
        origin=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        fov=55.0,
    )
    assert returned is scene

    structure = wr.Structure(
        name="target",
        geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(1.0, 1.0, 1.0)),
        material=wr.Material(eps_r=3.0),
    )
    vertices, faces = wr.Box(position=(0.0, 0.0, -2.0), size=(0.5, 0.5, 0.5)).to_mesh()

    assert scene.add_structure(structure) is scene
    assert scene.add_mesh(name="mesh", vertices=vertices, faces=faces) is scene
    assert scene.add_structure_motion("target", translation={"offset": (0.1, 0.0, 0.0)}) is scene

    assert scene.sensor.fov == 55.0
    assert [item.name for item in scene.structures] == ["target", "mesh"]
    assert "target" in scene._structure_motions
    assert not hasattr(scene, "with_sensor")
    assert not hasattr(scene, "with_structure")
    assert not hasattr(scene, "with_mesh")
    assert not hasattr(scene, "with_smpl")
    assert not hasattr(scene, "with_structure_motion")
