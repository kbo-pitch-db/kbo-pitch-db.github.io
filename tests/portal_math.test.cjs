const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const ctx={window:{}};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(__dirname,'../portal.js'),'utf8'),ctx);
const {jointAngle,transitionPairs}=ctx.window.PitchPortal;
const frame={keypoints:[[1,0,.9],[0,0,.9],[0,1,.9]]};
assert.equal(jointAngle(frame,[0,1,2]),90);frame.keypoints[2]=[-1,0,.9];assert.equal(jointAngle(frame,[0,1,2]),180);
for(const invalid of [null,[0,1,.49],[0,1,NaN],[0,1],['0',1,.9]]){frame.keypoints[2]=invalid;assert.equal(jointAngle(frame,[0,1,2]),null)}
frame.keypoints[2]=[0,1,.9];frame.interpolated=[false,true,false];assert.equal(jointAngle(frame,[0,1,2]),null);
const record=(gameId,atBat,sequence,atBatPitch,pitch='직구')=>({gameId,atBat,sequence,atBatPitch,pitch});
const records=[record('a','1',1,1),record('a','1',2,2),record('a','1',4,4),record('a','2',5,1),record('b','1',6,2),record('b','1',7,3)];
assert.equal(transitionPairs(records).length,2);
assert.equal(records[0].sequence,1);
assert.equal(transitionPairs([record('a','1',1,1),record('a','1',2,3)]).length,0);
assert.equal(transitionPairs([record('a','1',1,1,null),record('a','1',2,2)]).length,0);
console.log('PASS: projected joint angles exclude invalid, low-confidence, interpolated observations; transitions never bridge games, at-bats or missing pitches');
