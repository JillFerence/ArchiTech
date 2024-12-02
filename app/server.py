import aiohttp
import asyncio
import uvicorn
from fastai.vision.all import *
from pathlib import Path
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import nbformat
from nbconvert import PythonExporter
import base64

import import_ipynb

from gradcam import *
import torch

export_file_url = 'https://drive.google.com/uc?export=download&id=1J-xTZrMoE5Jfq90-OT2W3NsbzZvSrSLZ'
export_file_name = 'export.pkl'

classes = ['American craftsman style',
 'Bauhaus architecture',
 'Palladian architecture',
 'Deconstructivism',
 'Georgian architecture',
 'Romanesque architecture',
 'Greek Revival architecture',
 'American Foursquare architecture',
 'Byzantine architecture',
 'Postmodern architecture',
 'Art Nouveau architecture',
 'Art Deco architecture',
 'Russian Revival architecture',
 'Edwardian architecture',
 'Achaemenid architecture',
 'Novelty architecture',
 'Baroque architecture',
 'Colonial architecture',
 'Ancient Egyptian architecture',
 'Tudor Revival architecture',
 'Queen Anne architecture',
 'Chicago school architecture',
 'Gothic architecture',
 'International style',
 'Beaux-Arts architecture']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        #learn = load_learner(path, export_file_name)
        learn = load_learner(path/'export.pkl')
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = PILImage.create(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


@app.route('/detect', methods=['POST'])
async def detect(request):
    
    path = os.getcwd()
    '''
    print("Current working directory:", os.getcwd())
    notebook_path = os.path.join(path, 'src', 'architectural-style-recognition')
    print("path:", notebook_path)
    notebook = import_ipynb.import_module(notebook_path)
    
    #load_image = notebook_namespace['load_image_not_path']
    #GradCAM = notebook_namespace['GradCAM']
    #overlay_heatmap = notebook_namespace['overlay_heatmap_not_path']
    '''
    
    print("IMAGE")
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = PILImage.create(BytesIO(img_bytes))
    #prediction = learn.predict(img)[0]

    print("MODEL")
    model = models.resnet152(pretrained=False)
    model_path = os.path.join(path, 'app', 'models', 'stage-2-resnet152.pth')
    print("path:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    print("LOAD")
    print(img.shape)
    img = load_image_not_path(img)

    print("PREDICT")
    output = model(img)
    _, pred_class = output.max(dim=1)
    pred_class = pred_class.item()
    
    # Apply Grad-CAM
    print("GRADCAM")
    target_layer = model.layer4[2].conv3 
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(img, pred_class)

    print("HEATMAP")
    overlayed_img = overlay_heatmap_not_path(heatmap, img)

    buffered = BytesIO()
    overlayed_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return JSONResponse({'result': img_str})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
