from ICT_DeepFake.preprocess import *

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./pretrained/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    cfg['pretrain'] = False
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    save_idx = 0
    input_path = './DATASET/input'
    output_path = './DATASET/align_output'
    json_path = './DATASET/paths'
    video_list = os.listdir(input_path)
    paths = []
    for video in tqdm(video_list):
        video_capture = cv2.VideoCapture(os.path.join(input_path, video))
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 15 != 0:
                continue
            img_path = os.path.join(tar_path, str(save_idx)+'.png')
            find_face = get_face(net, frame, img_path)
            if find_face:
                save_idx += 1
                paths.append(img_path)
    json.dump(paths, open('./DATASET/paths/out.json', 'w'))