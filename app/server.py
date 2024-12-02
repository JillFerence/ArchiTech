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

import base64
from gradcam import *

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
    try:
        learn = load_learner(path/'models'/'export.pkl')
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

    # LOAD INPUT IMAGE
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = PILImage.create(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]

    # LOAD MODEL
    path = os.getcwd()
    model = models.resnet152(pretrained=False)
    model_path = os.path.join(path, 'app', 'models', 'stage-2-resnet152.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    # UPDATE IMAGE FORMAT
    img = load_image_not_path(img)

    # PREDICT
    output = model(img)
    _, pred_class = output.max(dim=1)
    pred_class = pred_class.item()

    # GRADCAM
    target_layer = model.layer4[2].conv3 
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(img, pred_class)

    # HEATMAP
    overlayed_img = overlay_heatmap_not_path(heatmap, img)
    overlayed_img_pil = Image.fromarray(overlayed_img)

    buffer = BytesIO()
    overlayed_img_pil.save(buffer, format="JPEG") 
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return JSONResponse({'result': str(prediction), 'image_data': img_base64})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
