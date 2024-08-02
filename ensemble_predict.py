from models.ensemble_classifier import EnsembleClassifier
from models.SSAM_ConvNeXt import ConvNeXt_FPN_Classifier
from models.SSAM_SWIN import SSAM_Swin_Classifier
from models.RCAN import RCAN_classifier
from utils.dataset import EnsembleDataset

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm.auto import tqdm

import torch
import pandas as pd



def inference(model, test_loader, device, save_path:Path, save_name, writer: SummaryWriter):
    model.eval()
    submission_file = {'Id':[],
                       'Category':[]}
    save_path.mkdir(parents=True,
                    exist_ok=True)

    save_path_file_name = save_path / f'submission_{save_name}.csv'

    with torch.inference_mode():
        pbar = tqdm(test_loader, desc='Test')
        for idx, (img_32, img_64) in enumerate(pbar):
            img_32 = img_32.to(device)
            img_64 = img_64.to(device)

            label_index = model(img_32, img_64)  # is index of labels already

            submission_file['Id'].append(str(test_loader.dataset.data_info[idx].name))
            submission_file['Category'].append(label_index.cpu().item()+1)


            if idx == 0:
                writer.add_graph(model=model,
                                 input_to_model=[img_32, img_64])

    csv_submission_file = pd.DataFrame(submission_file)

    csv_submission_file.to_csv(save_path_file_name,
                               index=False)

    print(f'Submission fils save to: {save_path_file_name}')


def main(save_path):
    save_path = Path(save_path)
    writer = SummaryWriter(log_dir=save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Models
    convnext_fpn_cls_pretrained_weight = 'weights/SSAM_ConvNeXt.pth'
    convnext_fpn_cls = ConvNeXt_FPN_Classifier()
    convnext_fpn_cls.load_state_dict(torch.load(convnext_fpn_cls_pretrained_weight,
                                                map_location='cpu'))

    ssam_swin_cls_pretrained_weight = 'weights/SSAM_SWIN.pth'
    ssam_swin_cls = SSAM_Swin_Classifier()
    ssam_swin_cls.load_state_dict(torch.load(ssam_swin_cls_pretrained_weight,
                                             map_location='cpu'))

    rcan_cls_pretrained_weight = 'weights/RCAN.pth'
    rcan_cls = RCAN_classifier()
    rcan_cls.load_state_dict(torch.load(rcan_cls_pretrained_weight,
                                        map_location='cpu'))

    ensemble_cls = EnsembleClassifier(convnext_fpn_cls=convnext_fpn_cls,
                                      ssam_swin_cls=ssam_swin_cls,
                                      rcan_cls=rcan_cls).to(device)


    # Dataset & Dataloader
    transform_convnext_swin = transforms.Compose([transforms.Resize(size=(64,64))])
    test_data = EnsembleDataset(data_dir="ICPR01/kaggle/evaluation", 
                                transform_convnext_swin=transform_convnext_swin)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)


    inference(ensemble_cls,
              test_loader,
              device,
              save_path=save_path,
              save_name='Ensemble',
              writer=writer)


if __name__ == '__main__':
    save_path = 'ensemble_cls_result'
    
    main(save_path)