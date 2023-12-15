window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];

function preloadInterpolationImages() {
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
        var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
        interp_images[i] = new Image();
        interp_images[i].src = path;
    }
}

function setInterpolationImage(i) {
    var image = interp_images[i];
    image.ondragstart = function () {
        return false;
    };
    image.oncontextmenu = function () {
        return false;
    };
    $('#interpolation-image-wrapper').empty().append(image);
}

document.addEventListener("DOMContentLoaded", (event) => {
    var b = document.querySelectorAll('.b-dics');
    b.forEach(element =>
        new Dics({
            container: element,
            textPosition: 'bottom',
            arrayBackgroundColorText: ['#000000', '#000000', '#000000'],
            arrayColorText: ['#FFFFFF', '#FFFFFF', '#FFFFFF'],
            linesColor: '#ffffff'
        })
    );

});

$(document).ready(function () {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function () {
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        $(".navbar-burger").toggleClass("is-active");
        $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
        slidesToScroll: 1,
        slidesToShow: 3,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for (var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
        // bulmaCarousel instance is available as element.bulmaCarousel
        element.bulmaCarousel.on('before-show', function (state) {
            console.log(state);
        });
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function (event) {
        setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

function changeRef(selected) {
    var im1 = document.getElementById("recolor_bonsai")
    var im2 = document.getElementById("recolor_room")
    var im3 = document.getElementById("recolor_horns")
    var im4 = document.getElementById("recolor_trex")
    if (selected.id == 'btn_pnf') {
        im1.src = 'static/comparison_pnf/bonsai/pnf_nosem.png'
        im2.src = 'static/comparison_pnf/room/pnf_nosem.png'
        im3.src = 'static/comparison_pnf/horns/pnf_nosem.png'
        im4.src = 'static/comparison_pnf/trex/pnf_nosem.png'
    }
    if (selected.id == 'btn_pnf_sem') {
        im1.src = 'static/comparison_pnf/bonsai/pnf.png'
        im2.src = 'static/comparison_pnf/room/pnf.png'
        im3.src = 'static/comparison_pnf/horns/pnf.png'
        im4.src = 'static/comparison_pnf/trex/pnf.png'
    }
}

var currentScene = 'horns';
var currentSceneRecolor = 'flower';
var currentSceneRecolorStyle = 'hornswave';

function changeScene(scene, style_base) {
    var activeContainer = document.getElementById(currentScene);
    activeContainer.className = 'is-hidden';

    // reset style for now non-selected
    var activeBtn = document.getElementById("btn_" + currentScene);
    activeBtn.className = 'button-17';

    currentScene = scene;
    //document.write(currentScene)
    var newContainer = document.getElementById(currentScene);
    newContainer.className = '';
    // set style for now selected
    var activeBtn = document.getElementById("btn_" + currentScene);
    activeBtn.className = 'button-17-selected';

    var im_ref = document.getElementById("gif_ref")
    im_ref.src = "static/demo/" + currentScene + '/' + currentScene + ".gif";

    var im = document.getElementById("gif_style");
    im.src = "static/demo/" + currentScene + '/' + style_base + ".gif";

    var circle = document.getElementById(currentScene + '_' + style_base);
    circle.className = "circle-selected";
}

function changeSceneRecolor(scene, style_base) {
    var activeContainer = document.getElementById('rec_' + currentSceneRecolor);
    activeContainer.className = 'is-hidden';

    // reset style for now non-selected
    var activeBtn = document.getElementById("recbtn_" + currentSceneRecolor);
    activeBtn.className = 'button-17';

    currentSceneRecolor = scene;
    //document.write(currentScene)
    var newContainer = document.getElementById('rec_' + currentSceneRecolor);
    newContainer.className = '';
    // set style for now selected
    var activeBtn = document.getElementById("recbtn_" + currentSceneRecolor);
    activeBtn.className = 'button-17-selected';

    var im_ref = document.getElementById("recgif_ref")
    im_ref.src = "static/demo/" + currentSceneRecolor + '/' + currentSceneRecolor + ".gif";

    var im = document.getElementById("recgif_style");
    im.src = "static/demo/" + currentSceneRecolor + '/' + style_base + ".gif";

    var circle = document.getElementById(currentSceneRecolor + '_' + style_base);
    circle.className = "circle-rec-selected";
}

function changeSceneRecolorStyle(scene, style_base) {
    var activeContainer = document.getElementById('rs_' + currentSceneRecolorStyle);
    activeContainer.className = 'is-hidden';

    // reset style for now non-selected
    var activeBtn = document.getElementById("rsbtn_" + currentSceneRecolorStyle);
    activeBtn.className = 'button-17';

    currentSceneRecolorStyle = scene;
    //document.write(currentScene)
    var newContainer = document.getElementById('rs_' + currentSceneRecolorStyle);
    newContainer.className = '';
    // set style for now selected
    var activeBtn = document.getElementById("rsbtn_" + currentSceneRecolorStyle);
    activeBtn.className = 'button-17-selected';

    var im_ref = document.getElementById("rsgif_ref")
    im_ref.src = "static/demo/" + currentSceneRecolorStyle + '/' + currentSceneRecolorStyle + ".gif";

    var im = document.getElementById("rsgif_style");
    im.src = "static/demo/" + currentSceneRecolorStyle + '/' + style_base + ".gif";

    var circle = document.getElementById(currentSceneRecolorStyle + '_' + style_base);
    circle.className = "circle-rs-selected";
}

function changeStyle(i) {
    var b = document.querySelectorAll('.button-17-selected');
    [].forEach.call(b, function (div) {
        // do whatever
        div.className = "button-17";
    });
    i.className = "button-17-selected";

    changeRef(i)
}

function changeGIF(style_image, image) {
    var b = document.querySelectorAll('.circle-selected');
    [].forEach.call(b, function (div) {
        // do whatever
        div.className = "circle";
    });
    image.className = "circle-selected";
    var im = document.getElementById("gif_style")
    var im_ref = document.getElementById("gif_ref")
    im.src = "static/demo/" + currentScene + '/' + style_image + ".gif";
    im_ref.src = im_ref.src;
}

function changeGIFrecolor(color, image) {
    var b = document.querySelectorAll('.circle-rec-selected');
    [].forEach.call(b, function (div) {
        // do whatever
        div.className = "circle-rec";
    });
    image.className = "circle-rec-selected";
    var im = document.getElementById("recgif_style")
    var im_ref = document.getElementById("recgif_ref")
    im.src = "static/demo/" + currentSceneRecolor + '/' + color + ".gif";
    im_ref.src = im_ref.src;
}

function changeGIFrecolorstyle(color, image) {
    var b = document.querySelectorAll('.circle-rs-selected');
    [].forEach.call(b, function (div) {
        // do whatever
        div.className = "circle-rs";
    });
    image.className = "circle-rs-selected";
    var im = document.getElementById("rsgif_style")
    var im_ref = document.getElementById("rsgif_ref")
    im.src = "static/demo/" + currentSceneRecolorStyle + '/' + color + ".gif";
    im_ref.src = im_ref.src;
}