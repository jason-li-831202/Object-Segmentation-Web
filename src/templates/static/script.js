
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function initSliderPosition(input_id){
  target = document.getElementById(input_id)

  /* alert("value :" + e.target.id) */;
  const min = target.min
  const max = target.max
  const val = target.value

  target.style.backgroundSize = (val - min) * 100 / (max - min) + '% 100%'

  return val;
}

function blockCapture() {
  html2canvas(document.querySelector("#capture")).then(function (canvas) {
      a = document.createElement('a');
      a.href = canvas.toDataURL("image/jpeg", 0.92).replace("image/jpeg", "image/octet-stream");
      a.download = 'screenshot.jpg';
      a.click();
  });
}

function sliderChangeHandler(e) {
  target = document.getElementById(e.target.id)

  const min = target.min
  const max = target.max
  const val = target.value
  
  target.style.backgroundSize = (val - min) * 100 / (max - min) + '% 100%'
}

function sliderClickHandler(e){
	let target = e.target
  // alert("id :" + target.id + "value :" + target.value);
  // Exposure switch
  if (target.id == "exposure") {
    $.get("/request_exposure", {'value': target.value}, function() {
      return
    });
  }else if (target.id == "contrast"){
    $.get("/request_contrast", {'value': target.value}, function() {
      return
    });
  }else if (target.id == "blur"){
    $.get("/request_blur", {'value': target.value}, function() {
      return
    });
  }

}


// Functions to deal with button events
$(function () {
  // Loading Video time
  // sleep(1000);
  
  // const spinner = document.getElementById("spinner");
  // videoElement.onload =function() {
  //   spinner.style.display = 'none'; 
  // }

  // Flask menu display
  $(".modelView").click(function () {
    $(".modelMenu").toggleClass("active");
    $(".chevron-right").toggleClass("rotate");
  });

  // Cam preview switch
  $("input#cam-preview").on('change',function(){
    if(this.checked) {
      // alert("Checked");
      $.get("/request_preview_switch", {'active': "True"}, function() {
				var img = document.getElementById("videoElement");
				img.style.display = "block";
        return
      });
    }
    else {
      // alert("No Checked");
      $.get("/request_preview_switch", {'active': "False"}, function() {
        var img = document.getElementById("videoElement");
        img.style.display = "none";
        return
      });
    }

  });

  // Upload background switch
  $("button#upload").bind("click", function () {
    const inputUrl = document.getElementById("video-url");
    const spinner = document.getElementById("spinner");
    try 
    {
      spinner.style.display = "block";
      const url = new URL(inputUrl.value);
      // alert("url path :" + inputUrl.value);
      $.get("/request_background_video", {'url': inputUrl.value}, function(response) {
        spinner.style.display = 'none'; 
        return
      });

      var btn = document.getElementById("background-preview");
      if (!btn.checked)
        btn.click(true);
    } 
    catch (error)
    {
      alert("URL parsing error, please check the path.");
    }
    
  });

  // Background switch
  $("input#background-preview").bind("click", function () {
    const inputUrl = document.getElementById("video-url");
    try 
    {
      const url = new URL(inputUrl.value);
      if(this.checked) {
        // alert("Checked");
        $.get("/request_background_switch", {'active': "True"}, function() {
          return
        });
      }
      else {
        // alert("No Checked");
        $.get("/request_background_switch", {'active': "False"}, function() {
          return
        });
      }
    } 
    catch (error)
    {
      this.checked = false;
      alert("URL parsing error, can't click it.");
    }

  });

  // Flip horizontal switch
  $("input#flip-horizontal").bind("click", function () {
    if(this.checked) {
      // alert("Checked");
      $.get("/request_flipH_switch", {'active': "True"}, function() {
        return
      });
    }
    else {
      // alert("No Checked");
      $.get("/request_flipH_switch", {'active': "False"}, function() {
        return
      });
    }

  });

  // Reset camera
  $("a#reset-cam").bind("click", function () {
    $.getJSON("/reset_camera", {'active': "False", 'type': "Semantic Mode"}, function (data) {
    });
    
    // init exposure
    var exposure = document.getElementById("exposure");
    exposure.value = default_exposure;
    var exposure_value = document.getElementById("exposure_value");
    exposure_value.value = exposure.value;
    initSliderPosition("exposure");
    // init contrast
    var contrast = document.getElementById("contrast");
    contrast.value = default_contrast;
    var contrast_value = document.getElementById("contrast_value");
    contrast_value.value = contrast.value;
    initSliderPosition("contrast");
    // init blur
    var blur = document.getElementById("blur");
    blur.value = default_blur;
    var blur_value = document.getElementById("blur_value");
    blur_value.value = blur.value;
    initSliderPosition("blur");
    // init flip switch
    var flip = document.getElementById("flip-horizontal"); 
    flip.checked = false;
    const selected = document.querySelector(".selected");
    selected.innerHTML = "Semantic Mode";
    return false;
  });

  // Use Target Listener
  var checkboxes = document.querySelectorAll("input[type=checkbox][name=target]");
  let targetList = new Array();
  checkboxes.forEach(function(checkbox) {
    checkbox.addEventListener('change', function() {
      targetList = 
        Array.from(checkboxes) // Convert checkboxes to an array to use filter and map.
        .filter(i => i.checked) // Use Array.filter to remove unchecked checkboxes.
        .map(i => i.value) // Use Array.map to extract only the checkbox values from the array of objects.
        
      // alert("type :" + typeof JSON.stringify(targetList));
      // alert("object :" + JSON.stringify(targetList));
      $.get("/request_target_display", {'targetList': JSON.stringify(targetList) }, function() {return});

    })
  });

  // Use Model Listener
  const displayModeContainer = document.querySelector('.options-container[name="display-mode"]');
  const displayModeSelected = document.querySelector('.selected[name="display-mode"]');
  const displayModeList = document.querySelectorAll('.option[name="display-mode"]');

  displayModeSelected.addEventListener("click", () => {
    displayModeContainer.classList.toggle("active");
  });

  displayModeList.forEach(o => {
    o.addEventListener("click", () => {
      displayModeSelected.innerHTML = o.querySelector("label").innerHTML;
      displayModeContainer.classList.remove("active");
      // alert("type :" + displayModeSelected.innerHTML);
      $.get("/request_model_switch", {'type': displayModeSelected.innerHTML}, function() {return});
    });
  });

  // Use Style Listener
  const displayStyleContainer = document.querySelector('.options-container[name="display-style"]');
  const displayStyleSelected = document.querySelector('.selected[name="display-style"]');
  const displayStyleList = document.querySelectorAll('.option[name="display-style"]');

  displayStyleSelected.addEventListener("click", () => {
    displayStyleContainer.classList.toggle("active");
  });

  displayStyleList.forEach(o => {
    o.addEventListener("click", () => {
      displayStyleSelected.innerHTML = o.querySelector("label").innerHTML;
      displayStyleContainer.classList.remove("active");
      // alert("type :" + displayStyleSelected.innerHTML);
      $.get("/request_style_switch", {'type': displayStyleSelected.innerHTML}, function() {return});
    });
  });

  // Use Slider Listener
  var default_exposure = initSliderPosition("exposure");
  var default_contrast = initSliderPosition("contrast");
  var default_blur = initSliderPosition("blur");
  const rangeInputs = document.querySelectorAll('input[type="range"]')
  const numberInput = document.querySelector('input[type="number"]')

  rangeInputs.forEach(input => {
    input.addEventListener('input', sliderChangeHandler)
    input.addEventListener('click', sliderClickHandler)
  })
  numberInput.addEventListener('input', sliderChangeHandler)

});

