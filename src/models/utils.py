from src.models.LSTMs.Three_Stage_Global.IMU2Leaf import IMU2Leaf_WheelPoser
from src.models.LSTMs.Three_Stage_Global.IMU2Leaf_FineTune import IMU2Leaf_WheelPoser_FineTune
from src.models.LSTMs.Three_Stage_Global.Leaf2Full import Leaf2Full_WheelPoser
from src.models.LSTMs.Three_Stage_Global.Leaf2Full_FineTune import Leaf2Full_WheelPoser_FineTune
from src.models.LSTMs.Three_Stage_Global.Full2Pose_Global import Full2Pose_WheelPoser_Global
from src.models.LSTMs.Three_Stage_Global.Full2Pose_Global_Finetune import Full2Pose_WheelPoser_Global_Finetune


def get_model(config=None, pretrained=None):
    model = config.model
    print(model)

    # #IMUPoser
    # if model == "IMUPoser_WheelPoser_AMASS":
    #     net = IMUPoser_WheelPoser (config = config)
    # elif model == "IMUPoser_WheelPoser_WHEELPOSER":
    #     net = IMUPoser_WheelPoser_FineTune (config = config, pretrained_model=pretrained)
    
    # #TIP
    # elif model == "TIP_WheelPoser_AMASS":
    #     net = TIP_WheelPoser (config = config)
    # elif model == "TIP_WheelPoser_WHEELPOSER":
    #     net = TIP_WheelPoser_Finetune (config = config, pretrained_model=pretrained)

    #IMU2Leaf
    if model == "IMU2Leaf_WheelPoser_AMASS":
        net = IMU2Leaf_WheelPoser (config = config)
    elif model == "IMU2Leaf_WheelPoser_WHEELPOSER":
        net = IMU2Leaf_WheelPoser_FineTune (config = config, pretrained_model=pretrained)

    #Leaf2Full
    elif model == "Leaf2Full_WheelPoser_AMASS":
        net = Leaf2Full_WheelPoser (config = config)
    elif model == "Leaf2Full_WheelPoser_WHEELPOSER":
        net = Leaf2Full_WheelPoser_FineTune (config = config, pretrained_model=pretrained)
    
    #Full2Pose
    elif model == "Full2Pose_WheelPoser_AMASS":
        net = Full2Pose_WheelPoser_Global (config = config)
    elif model == "Full2Pose_WheelPoser_WHEELPOSER":
        net = Full2Pose_WheelPoser_Global_Finetune (config = config, pretrained_model=pretrained)

    else:
        net = None

    return net