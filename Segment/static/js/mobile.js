init();
function init()
{
	document.body.style.overflow='hidden';//禁止滑动页面
}


function showresult()
{
	var windowHeight = $(window).height();
	var windowWidth = $(window).width();
	document.getElementById("loading").style.display="none";
	document.getElementById("resultdisplay").style.width=windowWidth+"px";
	document.getElementById("resultdisplay").style.height=windowWidth+"px";
	document.getElementById("result").style.height = windowHeight+"px";
	document.getElementById("resultdisplay").style.display="block";
	document.getElementById("info").style.display="block";
	document.getElementById("return").style.display="block";
	document.getElementById("resultimg").style.display="block";
}

document.getElementById("return").addEventListener('click',function(){
	document.getElementById("result").style.display="none";
	document.getElementById("camera").style.display="block";
	document.getElementById("resultimg").style.display="none";
	document.getElementById('tacklabel').style.display="block";
	$("#uploadlabel").css("display","block");
	$("#resultdisplay").css("display","none");
	document.getElementById("loading").style.display="block";
	document.getElementById("info").style.display="none";
	document.getElementById("return").style.display="none";
	$("#infobtn").css("display","block");
});




$.fn.spin = function(opts) {
    this.each(function() {
      var $this = $(this),
          data = $this.data();

      if (data.spinner) {
        data.spinner.stop();
        delete data.spinner;
      }
      if (opts != false) {
        data.spinner = new Spinner($.extend({color: $this.css('color')}, opts)).spin(this);
      }
    });
    return this;
  };

  function update() {
    var opts = {};
    $('#opts input[min]').each(function() {
      $('#opt-' + this.name).text(opts[this.name] = parseFloat(this.value));
    });
    $('#opts input:checkbox').each(function() {
      opts[this.name] = this.checked;
      $('#opt-' + this.name).text(this.checked);
    });
    $('#Loading_img').spin(opts);
  }
  update();


    // 选择文件触发事件  
    function selectImg(file) {  
        //文件为空，返回  
        if (!file.files || !file.files[0]) {  
            return;  
        }   
        var reader = new FileReader();
        var files = file.files[0];
        reader.readAsDataURL(file.files[0]);
        reader.onload = function(evt) {
            uploadFile(this.result);
        }
        //sleep(200).then(()=> {});
        //$("div[class$="cropper-wrap-box"]").eq(0).css("display","none");  

        
    }  


 //ajax请求上传
        function uploadFile(imgbase64) {
            //这里添加display:none

            document.getElementById("infobtn").style.display="none";

            document.getElementById("resultimg").style.display="block";
            document.getElementById("Loading_img").style.display="block";
            document.getElementById("result").style.display="block";
	        document.getElementById("camera").style.display="none";
	        document.getElementById("resultimg").style.display="block";
	        document.getElementById('tacklabel').style.display="none";

	        document.getElementById("loading").style.display="block";
	          document.getElementById("info").style.display="block";

            $.ajax({
            	url: 'Segment/forecast/ol_update',     //请求url地址
            	type: "POST",
            	data: {"imgbase64":imgbase64},       //发送post请求携带的数据信息
            	success: function (data)
            	{
            	    imgfirst = document.getElementById("temp").src;
                	data = JSON.parse(data);
                	//forecast_results = data[0].results;
                	//document.getElementById("message").innerHTML=forecast_results;
                	console.log(data);
                	document.getElementById("resultimg").src = imgfirst+data[0];
//                	console.log(data[0])
             		showresult();
             }
     		});
        }




  