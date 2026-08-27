def make_static_fr3_duo_config():
    C = ry.Config()

    table = C.addFrame("table").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(1)

    fr3_path = os.path.join(os.path.dirname(__file__), "../../assets/models/rai/fr3/static_mobile_fr3_base.g")
    
    fr3_left_path = os.path.join(os.path.dirname(__file__), "../../assets/models/rai/fr3/left_fr3.g")
    fr3_right_path = os.path.join(os.path.dirname(__file__), "../../assets/models/rai/fr3/right_fr3.g")

    C.addFile(fr3_path, namePrefix="fr3_")

    right = C.addFile(fr3_right_path, namePrefix="a1_").setParent(C.getFrame("fr3_right_base_joint_origin"))
    left = C.addFile(fr3_left_path, namePrefix="a2_").setParent(C.getFrame("fr3_left_base_joint_origin"))

    right_ee = (
        C.addFrame("right_ee")
        .setParent(C.getFrame("a1_right_fr3v2_link8"))
        .setShape(ry.ST.marker, size=[0.5])
        .setColor([1, 0.5, 0])
        .setContact(0)
    )
    
    right_ee = (
        C.addFrame("left_ee")
        .setParent(C.getFrame("a2_left_fr3v2_link8"))
        .setShape(ry.ST.marker, size=[0.5])
        .setColor([1, 0.5, 0])
        .setContact(0)
    )
    
    height = 0.5