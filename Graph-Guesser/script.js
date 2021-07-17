class Graph {
  constructor(data,xlabel,ylabel){
    this.data = data;
    this.xlabel = xlabel;
    this.ylabel = ylabel;
  }

  static range_lerp(n,a1,b1,a2,b2){
    return (n-a1) / (b1-a1) * (b2-a2) + a2
  }

  static shapes(){
    return ["increase","decrease","random","resonance","parabolic"]
  }

  y_from(x){
    return this.data.filter(e=>e.x==x).map(e=>e.y)[0]
  }

  display(surface,x,y,width,height){
    const x_cords = this.data.map((e,i)=>i).map(e=>Graph.range_lerp(e,0,this.data.length-1,x,x+width))
    const y_cords = this.data.map((e,i)=>e.y).map(e=>Graph.range_lerp(e,Math.min(...this.data.map(e=>e.y)),Math.max(...this.data.map(e=>e.y)),y,y-height))

    surface.save();
    surface.beginPath();
    surface.lineWidth = 5;
    surface.moveTo(x,y);
    surface.lineTo(x+width,y);
    surface.moveTo(x,y);
    surface.lineTo(x,y-height);
    surface.stroke()

    
    surface.save();
    surface.lineWidth = 2

    for (let i=1; i<x_cords.length;i++){
      surface.beginPath();
      surface.moveTo(x_cords[i-1],y_cords[i-1]);
      surface.lineTo(x_cords[i],y_cords[i]);
      surface.stroke();

      surface.fillStyle = "red"
      surface.beginPath();
      surface.ellipse(x_cords[i],y,2,2,0,0,Math.PI*2);
      surface.fill();
      
      surface.fillStyle = "blue"
      surface.beginPath();
      surface.ellipse(x,y_cords[i],2,2,0,0,Math.PI*2);
      surface.fill();
      
      surface.fillStyle = "green"
      surface.beginPath();
      surface.ellipse(x_cords[i],y_cords[i],5,5,0,0,Math.PI*2)
      surface.fill();

    }

    surface.fillStyle = "green"
    surface.beginPath();
    surface.ellipse(x_cords[0],y_cords[0],5,5,0,0,Math.PI*2)
    surface.fill();

    surface.restore();
  }
}

function createData(x,shape){
  const noise = (mag,prob) =>  Math.random() * mag * (Math.random()>=prob ? 1 : -1);
  if (shape=="increase"){
    return x.map(e=> e + noise(5,0.5))
  } else if (shape=="decrease") {
    return x.map(e=> -e + noise(5,0.5))
  } else if (shape=="resonance"){
    return x.map((e,i)=> ((i%2 ? 1 : -1) + noise(0.2,0.5)) * e)
  } else if (shape=="parabolic"){
    return x.map(e=> e**2 + noise(5,0.5))
  } else if (shape=="random"){
    return x.map(e=> Math.random() * x.length)
  }
}

class GraphShapeClassifier {
  constructor(graph,nn=null){
    this.graph = graph
    this.stopTraining = false
    if (nn){
      this.sequential = nn;
      this.sequential.compile({loss:"meanSquaredError",optimizer:"adam"})
    }else {
      this.sequential = tf.sequential();
      this.sequential.add(tf.layers.dense({units:20,batchInputShape:[null,this.graph.data.length*2],activation:"sigmoid"}))
      this.sequential.add(tf.layers.dense({units:20,activation:"sigmoid"}))
      this.sequential.add(tf.layers.dense({units:5,activation:"softmax"}))
      this.sequential.compile({loss:"meanSquaredError",optimizer:"adam"})
    }
  }

  classify(){
    const input = tf.tensor2d([this.graph.data.map(e=>e.x).concat(this.graph.data.map(e=>e.y))])
    const result = this.sequential.predict(input)
    input.dispose();
    //result.print();
    let ret = result.arraySync();
    result.dispose();
    return ret.map(ep=>ep.map((e,i)=>[e,Graph.shapes()[i]]).sort((a,b)=>b[0]-a[0]));
  }

  async train(trainingSet,iter){
    for (let i=0; i < iter; i++){
      if (this.stopTraining){
        this.stopTraining = false;
        break;
      }
      const inputs = tf.tensor2d(trainingSet.inputs);
      const outputs = tf.tensor2d(trainingSet.outputs);

      const h = await this.sequential.fit(inputs,outputs,{epochs:10,batchSize:10,shuffle:true})
      //console.log(h)
      inputs.dispose();
      outputs.dispose();
      console.log("Training completed: "+(i/iter*100).toString()+"%"," Current loss: " + h.history.loss[0].toString())
	  //console.log("Memory: ",tf.memory().numTensors)
    }
    return 0
  }
}

function createDataSet(n,n2){
  const res = {inputs:[],outputs:[]};
  const constantPart = Array(n2).fill().map((e,i)=>(i+1))
  for (let i=0; i < n; i++){
    let cs = Graph.shapes()[Math.floor(Math.random() * Graph.shapes().length)]
    res.inputs.push(constantPart.concat(createData(constantPart,cs)))
    res.outputs.push(Graph.shapes().map(e=>(e==cs)+0))
  }
  return res;
}

/*Html stuff*/
window.onload = _ => {
Graph.shapes().forEach( s => {
  let btn = document.createElement("input");
  btn.type = "button";
  btn.value = s;
  btn.class = "btn";
  btn.onclick = e => {
    g = new Graph(my_x.map((e,i)=>({x:e,y:createData(my_x,btn.value)[i]})),"Time","Money");
    k.graph = g;
  }
  document.getElementById("buttons").appendChild(btn)
})

}
/* -- */

let RUN = true
const my_x = Array(100).fill().map((e,i)=>(i+1))
const my_y = createData(my_x,"resonance");

let g = new Graph(my_x.map((e,i)=>({x:e,y:my_y[i]})),"Time","Money")
let k = new GraphShapeClassifier(g,null);

const screen = document.getElementById("screen");
const aiScreen = document.getElementById("aiScreen");
const trainingSet = createDataSet(100,my_x.length)
//g.display(screen.getContext("2d"),8,screen.height-8,screen.width,screen.height)

function displayAiPrediction(prediction){
  const arr = Graph.shapes();
  const ctx = aiScreen.getContext("2d");
  ctx.save();
  ctx.fillStyle = "green";
  ctx.beginPath();
  ctx.ellipse(aiScreen.width / 2 , aiScreen.height / 5 ,5,5,0,0,Math.PI*2);
  ctx.fill();
  ctx.restore();
  for (let i=0; i < arr.length; i++){
    ctx.beginPath();
    ctx.ellipse(i * aiScreen.width / arr.length + (aiScreen.width / arr.length / 2) ,aiScreen.height / 5 * 3,5,5,0,0,Math.PI*2);
    ctx.fill();
  }
  ctx.save();
  ctx.strokeStyle = "red"
  ctx.beginPath();
  ctx.moveTo(aiScreen.width / 2, aiScreen.height / 5)
  ctx.lineTo(arr.indexOf(prediction) * aiScreen.width / arr.length + (aiScreen.width / arr.length / 2) ,aiScreen.height / 5 * 3)
  ctx.stroke()
  ctx.restore();
  return 0
}

function mainLoop(){
  screen.getContext("2d").clearRect(0,0,screen.width,screen.height);
  aiScreen.getContext("2d").clearRect(0,0,aiScreen.width,aiScreen.height);
  g.display(screen.getContext("2d"),8,screen.height-8,screen.width,screen.height)

  displayAiPrediction(k.classify()[0][0][1]);
  if(RUN) requestAnimationFrame(mainLoop);
}

mainLoop();


function noLoop(){
  RUN = false;
}

function loop(){
  RUN = true
  mainLoop();
}

async function trainAI(){
  let temp = document.createElement("input")
  temp.type = "button"
  temp.value = "Stop training"
  temp.onclick = e => k.stopTraining = true;
  document.body.appendChild(temp);
  await k.train(createDataSet(100,my_x.length),100)
  document.body.removeChild(temp);
}

async function exportAI(){
  noLoop();
  await k.sequential.save(`downloads://model${Math.floor(Math.random() * 10000)}`)
  console.log("Successfully exported!")
}


async function importAI(){
  const model = await tf.loadLayersModel("Trained Network/graphGuessModel.json")
  k = new GraphShapeClassifier(g,model);
  console.log("Successfully imported!")
}

//console.log(trainingSet)
//console.log("Pre-training: ",k.classify())
//k.train(trainingSet,100).then(d=>console.log("Post-training: ",k.classify()))
//console.log(tf.memory().numTensors)
