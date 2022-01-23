//import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4
import links from './links';
import { update } from '@tensorflow/tfjs-layers/dist/variables';
import { div } from '@tensorflow/tfjs';
let imageCapture;
var stop= new Boolean(false);
var blur= new Boolean(true);
var toques=0;
var timer; 
var time=4;
var rand;
var style_img;
let engine1 = null;
let engine2 = null;
let engine3 = null;
let engine4 = null;

let faceapi, faceapi2,faceapi3;
let detections;

let vid_front, vid_left, vid_right;
const width = 1280;
const height = 720;

//ID's Cameras //EDIT HERE YOUR DEVICES IDS
let id_right_cam = "500507536bbb1fa5a2273d3dc1a6f0225db7f4ebce979406fef3f1b5f31d3c91";
let id_front_cam = "c901ec76ec5aa2bb3f9f41001a9d8ce8be44fe968a8973ca8caf16bb098b57fd";
let id_left_cam = "8b7ec796395ff4039919b25bd4788d12028d473b8f9f4b1e12756d1b82492bfb";

let canvas, ctx;
let canvas_pres, ctx_pres;
let canvas_past, ctx_past;
let canvas_fut, ctx_fut;

//Cameras Initialization
var videodevicesid = new Array();
var cam1, cam2, cam3;
var cameras = new Array();

var divs_pres = document.getElementsByClassName('pres');
var divs_past = document.getElementsByClassName('past');
var divs_fut = document.getElementsByClassName('fut');

var partes = ['face', 'nose', 'mouth', 'leftEye', 'rightEye'];
var style_images = ['style1.jpeg', 'style2.jpeg', 'style3.jpeg', 'style4.jpeg', 'style5.jpeg', 'style6.jpeg'];

//Canvas Capture Past + Rendering 2D
var all_canvas_past = new Array();
var all_ctx_past = new Array();
//Canvas Capture Present + Rendering 2D
var all_canvas_pres = new Array();
var all_ctx_pres = new Array();
//Canvas Capture Future + Rendering 2D
var all_canvas_fut = new Array();
var all_ctx_fut = new Array();

// by default all options are set to true
const detectionOptions = {
  withLandmarks: true,
  withDescriptors: false,
};


document.getElementById('button1').addEventListener('click', function (event) {
  document.getElementById('main').style.display = 'none';
  document.getElementById('canvasContainer').style.display = 'grid';
  // do something

  //Get Video inputs ID'S

  navigator.mediaDevices.enumerateDevices()
    .then(function (devices) {
      devices.forEach(function (device,index) {
        var id = device.deviceId;
        
        if (device.kind === 'videoinput') {
          console.log('Cameras IDs:  ' + id);

          if (id === id_front_cam) {
            cam1 = id;
            cameras.push(cam1);
          } else if (id === id_left_cam) {
            cam2 = id;
            cameras.push(cam2);
          } else if (id === id_right_cam) {
            cam3 = id;
            cameras.push(cam3);
          }
        }
      });
      return [cam1, cam2, cam3];
    }).then(function (cam) {
      make(cam);
    })

});

/**
 * Main application to start on window load
 */
class Main {
  constructor(index) {
    if (window.mobilecheck()) {
      document.getElementById('mobile-warning').hidden = false;
    }

    this.loadInceptionStyleModel().then(model => {
      this.styleNet = model;
    }).finally(() => this.startStyling());

    this.loadSeparableTransformerModel().then(model => {
      this.transformNet = model;
    }).finally(() => this.startStyling());

    this.initializeStyleTransfer(index);


    Promise.all([
      this.loadInceptionStyleModel(),
      this.loadSeparableTransformerModel(),
    ]).then(([styleNet, transformNet]) => {
      console.log('Loaded styleNet');
      this.styleNet = styleNet;
      this.transformNet = transformNet;
      //this.startStyling();
    });
  }

  async loadInceptionStyleModel() {
    if (!this.inceptionStyleNet) {
      this.inceptionStyleNet = await tf.loadGraphModel(
        'saved_model_style_inception_js/model.json');
    }

    return this.inceptionStyleNet;
  }

  async loadSeparableTransformerModel() {
    if (!this.separableTransformNet) {
      this.separableTransformNet = await tf.loadGraphModel(
        'saved_model_transformer_separable_js/model.json'
      );
    }

    return this.separableTransformNet;
  }

  
  initializeStyleTransfer(index) {
      this.contentImg = canvas_fut[0][index];
      

      this.contentImg.onerror = () => {
        alert("Error loading " + this.contentImg.src + ".");
      }
    

      this.styleImg = document.getElementById('style-img');
      this.styleImg.onerror = () => {
        alert("Error loading " + this.styleImg.src + ".");
      }

      //div onde vai desenhar
      var div_draw = document.getElementById('f'+index.toString());
      this.stylized = div_draw.firstChild;
      this.stylized_ctx = this.stylized.getContext('2d');
      this.stylized_w = this.stylized.offsetWidth;
      this.stylized_h= this.stylized.offsetHeight;
      this.styleRatio = 1.0;
  }

  async startStyling() {
      await tf.nextFrame();
      console.log('Starting styling...');

      await tf.nextFrame();
      let bottleneck = await tf.tidy(() => {
        return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
      })
      if (this.styleRatio !== 1.0) {
        console.log('One more thing...');

        await tf.nextFrame();
        const identityBottleneck = await tf.tidy(() => {
          return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
        })
        const styleBottleneck = bottleneck;
        bottleneck = await tf.tidy(() => {
          const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(this.styleRatio));
          const identityBottleneckScaled = identityBottleneck.mul(tf.scalar(1.0 - this.styleRatio));
          return styleBottleneckScaled.addStrict(identityBottleneckScaled)
        })
        styleBottleneck.dispose();
        identityBottleneck.dispose();
      }
      console.log('Getting Ready...');
      await tf.nextFrame();
      const stylized = await tf.tidy(() => {
        return this.transformNet.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
      })

      await tf.browser.toPixels(stylized, this.stylized);
      //bottleneck.dispose();  // Might wanna keep this around
      this.stylized_ctx.drawImage(this.stylized, 10, 10, this.stylized_w - 20 , this.stylized_h - 25 , 0, 0, this.stylized_w, this.stylized_h);
      console.log(this.stylized.offsetWidth);
      stylized.dispose();
  }

  async benchmark() {
    const x = tf.randomNormal([1, 256, 256, 3]);
    const bottleneck = tf.randomNormal([1, 1, 1, 100]);

    let styleNet = await this.loadInceptionStyleModel();
    let time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    styleNet = await this.loadMobileNetStyleModel();
    time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    let transformNet = await this.loadOriginalTransformerModel();
    time = await this.benchmarkTransform(
      x, bottleneck, transformNet);
    transformNet.dispose();

    transformNet = await this.loadSeparableTransformerModel();
    time = await this.benchmarkTransform(
      x, bottleneck, transformNet);
    transformNet.dispose();

    x.dispose();
    bottleneck.dispose();
  }

  async benchmarkStyle(x, styleNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = styleNet.predict(x);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = styleNet.predict(x);
          y.print();
        }
      })
    });
    console.log(time);
  }

  async benchmarkTransform(x, bottleneck, transformNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = transformNet.predict([x, bottleneck]);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = transformNet.predict([x, bottleneck]);
          y.print();
        }
      })
    });
    console.log(time);
  }
}

window.mobilecheck = function () {
  var check = false;
  (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
  return check;
};



document.addEventListener("keydown", function(e) {
  if (e.key === " ") {
    toques++;
    if((toques%2) != 0){
      blur = false;
      timer = setInterval(countDown, 1500);
    } else {
      //Clear Timer
      document.getElementById("timer").innerHTML = '';
      clearInterval(timer);
      time=3;
      //Run Style Transfer for all Divs_Fut
      engine1 = null;
      engine2 = null;
      engine3 = null;
      engine4 = null;
      blur = true;
      stop = false;

      rand = Math.round(Math.random(0,6));
      style_img = document.getElementById('style-img');

      style_img.src = '/images/' + style_images[rand];

      console.log(rand);
    } 
  }
}, false);

//Set CountDown Before StaticVideo
function countDown(){ 
  if(time<=4 && time>1){
  time--;
  document.getElementById("timer").innerHTML = time.toString();
 
  } else if (time==1){
    //Desactivate Motion Blur
    for(let i=0; i<all_ctx_fut.length;i++){
      all_ctx_fut[i].globalAlpha = 1;
    }

    clearInterval(timer);
    document.getElementById("timer").innerHTML = '';
    stop = true;
    runStyleTransfer();
  }
};

function runStyleTransfer(){
  if(engine1 == null && engine2 == null && engine3 == null && engine4 == null){
      engine1 = new Main(1);
      engine2 = new Main(2);
      engine3 = new Main(3);
      engine4 = new Main(4);
  }
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
}

async function make(cam) {
 //Choses a random source image for the Style Image
 rand = getRandomInt(0, 7);
 style_img = document.getElementById('style-img');

 style_img.src = '/images/' + style_images[1];

 console.log(rand);
  
  // get the video
  //Front Camera - Present
  vid_front = await getVideo(cam[0], divs_pres);
  canvas_pres = createCanvas(divs_pres);
  console.log("Ready! Cam Front!");

  //Left Camera - Past
  vid_left = await getVideo(cam[1], divs_past);
  canvas_past = createCanvas(divs_past);
  console.log("Ready! Cam Left");

  //Right Camera - Fut
  vid_right = await getVideo(cam[2], divs_fut);
  canvas_fut = createCanvas(divs_fut);
  console.log("Ready! Cam Right");

  faceapi = ml5.faceApi(vid_front, detectionOptions, modelReady1);
  faceapi2 = ml5.faceApi(vid_left, detectionOptions, modelReady2);
  faceapi3 = ml5.faceApi(vid_right, detectionOptions, modelReady3);
}

function modelReady1() {
  console.log("Loading...");
  faceapi.detect(gotResults1);
}

function modelReady2() {
  console.log("Loading...");
  faceapi2.detect(gotResults2);
}

function modelReady3() {
  console.log("Loading...");
  faceapi2.detect(gotResults3);
}

function gotResults1(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  detections = result;

  //canvas_pres[0] is all_canvas_pres
  //canvas_pres[1] is all_ctx_pres
  canvas = canvas_pres[0];
  ctx = canvas_pres[1]

  var canvas_w = canvas[0].offsetWidth;
  var canvas_h = canvas[0].offsetHeight;
  var vid_w = vid_front.width;
  var vid_h = vid_front.height;

  // Clear part of the canvas
  // Draws video on first canvas created located on div id=p0
  ctx[0].fillStyle = "#000000";
  ctx[0].fillRect(0, 0, canvas_w, canvas_h);

  ctx[0].drawImage(vid_front, 0, 0, width, height);

  if (detections && stop==false) {
    if (detections.length > 0) {
      drawLandmarks(detections, ctx, canvas);
    }
  }
  faceapi.detect(gotResults1);

}

function gotResults2(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  detections = result;

  //Canvas_pres[0] is all_canvas_pres
  //Canvas_pres[1] is all_ctx_pres
  canvas = canvas_past[0];
  ctx = canvas_past[1];

  var canvas_w = canvas[0].offsetWidth;
  var canvas_h = canvas[0].offsetHeight;
  var vid_w = vid_left.width;
  var vid_h = vid_left.height;

  // Clear part of the canvas
  //Draws video on first canvas created located on div id=p0
  ctx[0].fillStyle = "#000000";
  ctx[0].fillRect(0, 0, canvas_w, canvas_h); 

  ctx[0].drawImage(vid_left, 0, 0, width, height);

  if (detections && stop==false) {
    if (detections.length > 0) {
      drawLandmarks(detections, ctx, canvas);
    }
  }
  faceapi2.detect(gotResults2);
}

function gotResults3(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  detections = result;

  //canvas_pres[0] is all_canvas_pres
  //canvas_pres[1] is all_ctx_pres
  canvas = canvas_fut[0];
  ctx = canvas_fut[1]

  var canvas_w = canvas[0].offsetWidth;
  var canvas_h = canvas[0].offsetHeight;
  var vid_w = vid_front.width;
  var vid_h = vid_front.height;

  // Clear part of the canvas
  // Draws video on first canvas created located on div id=p0
  ctx[0].fillStyle = "#000000";
  ctx[0].fillRect(0, 0, canvas_w, canvas_h);

  ctx[0].drawImage(vid_right, 0, 0, width, height);

  if (detections && stop==false) {
    if (detections.length > 0) {
      drawLandmarks(detections, ctx, canvas);
    }
  }
  faceapi3.detect(gotResults3);

}

function drawLandmarks(detections, ctx, canvas) {
  for (let i = 0; i < detections.length; i += 1) {
    const mouth = detections[i].parts.mouth;
    const nose = detections[i].parts.nose;
    const leftEye = detections[i].parts.leftEye;
    const rightEye = detections[i].parts.rightEye;

    //----NOSE
    var nose_x = nose[4]._x;
    var nose_y = nose[0]._y - 10;
    var nose_w = 140;
    var nose_h = 150;

    //----MOUTH
    var mouth_x = mouth[0]._x-10;
    var mouth_y = mouth[4]._y - 30;
    var mouth_w = 145;
    var mouth_h = 137;

    //-----OLHO ESQUERDO
    var leftEye_x = leftEye[0]._x;
    var leftEye_y = leftEye[0]._y - 15;
    var leftEye_w = 135;
    var leftEye_h = 130;

    //-----OLHO DIR
    var rightEye_x = rightEye[0]._x;
    var rightEye_y = rightEye[0]._y - 15;
    var rightEye_w = 135;
    var rightEye_h = 130;

    
    
    drawPart(1, canvas, ctx, nose_x, nose_y, nose_w, nose_h);
    drawPart(2, canvas, ctx, mouth_x, mouth_y, mouth_w, mouth_h);
    drawPart(3, canvas, ctx, leftEye_x, leftEye_y, leftEye_w, leftEye_h);
    drawPart(4, canvas, ctx, rightEye_x, rightEye_y, rightEye_w, rightEye_h);

  }
}

function drawPart(index, canvas, ctx, x, y, larg, alt) {

  ctx[index].fillStyle = "#000000";
  ctx[index].fillRect(0, 0, canvas[index].width, canvas[index].height);

  //Get width from section
  let id_section = canvas[index].parentElement.id;
  var section = document.getElementById(id_section);

  var section_w = section.offsetWidth;
  var section_h = section.offsetHeight;
  
      if((index === 1 && section.className != 'fut') || id_section==='pt3' || id_section==='p4' || id_section==='p3' || id_section==='p2') {
        ctx[index].drawImage(canvas[0], x, y, larg, alt, 0, 0,  width*(section_h/width), section_h);
      
      } else if(section.className === 'fut'){
        if(index===1 || index ===4){
          ctx[index].drawImage(canvas[0], x-50, y, larg+100, alt+100, 0, 0,  width*(section_h/width), section_h);
        } else if (index===2){
          ctx[index].drawImage(canvas[0], x, y-20, larg+50, alt+50, 0, 0, section_w, height*(section_w/height)-5);
        } else if(index===3){
          ctx[index].drawImage(canvas[0], x, y, larg+50, alt+50, 0, 0, section_w, height*(section_w/height)-5);
        }
        if(blur==true){
          ctx[index].globalAlpha = 0.2;
        } else {
          ctx[index].globalAlpha = 1;
        }
        
      } else {
        ctx[index].drawImage(canvas[0], x, y, larg, alt, 0, 0, section_w, height*(section_w/height));
      } 
}

// Helper Functions
async function getVideo(cam, div) {
  //Constrains camera
  let constrains = {
    audio: false,
    video: {
      deviceId: { exact: cam }
    }
  };

  // Grab elements, create settings, etc.
  const videoElement = document.createElement("video");
  videoElement.setAttribute("style", "display: none;");
  videoElement.setAttribute("id", "video" + div[0].className);

  //Append To Video to First Div
  div[0].appendChild(videoElement);

  videoElement.width = width;
  videoElement.height = height;

  // Create a webcam capture
  const capture = await navigator.mediaDevices.getUserMedia(constrains);
  videoElement.srcObject = capture;

  // Create a webcam capture
  videoElement.play();
  if (div[0].className === 'fut') {
    const track = capture.getVideoTracks()[0];
    imageCapture = new ImageCapture(track);
    setTimeout(function () {
     // getFrame();
    }, 1000);
  }

  return videoElement;
}

function createCanvas(div) {
  for (let i = 0; i < div.length; i++) {

    //Create canvas element in each <div></div>
    //Gives an id like = pres_nariz (div present w/nose section)
    const canvas = document.createElement("canvas");
    canvas.setAttribute('id', div[i].className + '_' + partes[i]);
    div[i].appendChild(canvas);

    //Get id of each section on the CSS Grid 
    //To use as width/heigh of canvas created
    var section_id = div[i].id;
    const section = document.getElementById(section_id);

    let width_div = section.offsetWidth;
    let heigh_div = section.offsetHeight;
    canvas.width = width_div;
    canvas.height = heigh_div;
    //Get Rendering Context 2D of each div
    ctx = canvas.getContext("2d");

    //Depending on the div it gives push to array previously created
    //All_canvas_past/pres = all element canvas created
    //All_ctx_past/pres = the rendering 2D of all canvas
    if (div === divs_past) {
      all_canvas_past.push(canvas);
      all_ctx_past.push(ctx);
    } else if (div === divs_pres) {
      all_canvas_pres.push(canvas);
      all_ctx_pres.push(ctx);
    } else if (div === divs_fut) {
      all_canvas_fut.push(canvas);
      all_ctx_fut.push(ctx);

    }
  }

  //returns the arrays depending on input div 
  if (div === divs_past) {
    return [all_canvas_past, all_ctx_past];
  } else if (div === divs_pres) {
    return [all_canvas_pres, all_ctx_pres];
  } else if (div === divs_fut) {
    return [all_canvas_fut, all_ctx_fut];
  }
}
