/* Standalone development sanitizer harness; build separately from the library. */
#include <assert.h>
#include <string.h>
#include "batch_sha256.c"
int main(void){
unsigned char key[32]={0}, data[1]={0}, output[64];
size_t offsets[2]={0,0};
unsigned char expected[32]={170,120,85,225,56,57,221,118,124,213,218,124,31,245,3,101,64,201,38,75,122,128,48,41,49,94,85,55,82,135,180,175};
for(int i=0;i<10000;i++){
memset(output,0xA5,sizeof output);
assert(keyprint_prf_batch(key,32,data,0,data,0,offsets,2,1,output+16,32)==1);
assert(memcmp(output+16,expected,32)==0);
for(int j=0;j<16;j++){assert(output[j]==0xA5);assert(output[j+48]==0xA5);}
}
assert(keyprint_prf_batch(NULL,32,data,0,data,0,offsets,2,1,output+16,32)==0);
assert(keyprint_prf_batch(key,32,data,0,data,0,offsets,2,1,output+16,33)==0);
offsets[1]=(size_t)-1;assert(keyprint_prf_batch(key,32,data,0,data,0,offsets,2,1,output+16,32)==0);
return 0;
}
