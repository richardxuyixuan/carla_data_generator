from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from data_utils import objects_filter

def main():
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg)
    dtsave = DataSave(cfg)
    try:
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()
        step = 0
        num_frame = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        NUM_FRAMES = cfg["SAVE_CONFIG"]["NUM_FRAMES"]
        while True:
            if step % STEP ==0:
                data = model.tick()
                data = objects_filter(data)
                dtsave.save_training_files(data)
                print(step / STEP)
                num_frame += 1
            else:
                model.world.tick()
            step+=1
            if num_frame == NUM_FRAMES:
                break
    finally:
        model.setting_recover()


if __name__ == '__main__':
    main()
