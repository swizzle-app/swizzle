 // set css transitions after page is loaded
 $(document).ready(function() {
    $("#nav-content").css("transition", "margin-left 0.45s cubic-bezier(0.645, 0.045, 0.355, 1)");
    $("#burger-top").css("transition", "all 0.25s ease-in-out");
    $("#burger-mid").css("transition", "all 0.25s ease-in-out");
    $("#burger-bottom").css("transition", "all 0.25s ease-in-out");
    $("#slider").css("transition", "0.35s cubic-bezier(0.645, 0.045, 0.355, 1)");
    $("#lbl_text").css("transition", "transition: all 0.1s linear");
    $("#back").css("transition", "all 0.25s ease-in-out");

    $("#dm_select").val($("body").hasClass("dark").toString());
 });
 
 // get chosen file to display filename
 $("#uploader").change(function() {
    let filename = this.files[0].name;
    $("#dd_text").text(filename);
});
// animate menu
$(document).on("click", function(event) {
    if (!$(event.target).closest('#nav').length) {
        $("#nav-burger").removeClass("burger-anim");
        $("#nav-content").removeClass("nav-slide-in");
    }
});
$("#nav-burger").on("click", function(event) {
    if ($("#nav-burger").hasClass('burger-anim')) {
        $("#nav-burger").removeClass("burger-anim");
        $("#nav-content").removeClass("nav-slide-in");
    } else {
        $("#nav-burger").addClass("burger-anim");
        $("#nav-content").addClass("nav-slide-in");
    }
});
// toggle darkmode
$("#dm_cb").on("click", function(event) {
    $("body").toggleClass("dark");
    $("#dm_select").val($("body").hasClass("dark").toString());
});
// check file before submitting
$("#swizzle_form").submit(function(event) {
    $("#msg").html("");
    let file = $("#uploader");
    let f = file.prop('files').length;
    if (f > 0) {
        $("#submit").css("display", "none");
        $("#msg").fadeIn().html("\
        <!-- loading animation -->\
        <div class='spinner' id='spinner'>\
            <div class='spinner_anim'>\
            <?xml version='1.0' encoding='utf-8'?>\
            <svg version='1.1' id='Ebene_1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' x='0px' y='0px'\
                viewBox='0 0 1000 500' style='enable-background:new 0 0 1000 500;' xml:space='preserve'>\
                <style type='text/css'>\
                    .spinnerWave{fill:none;stroke:#7900AA;stroke-width:2.8402;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;}\
                </style>\
                <g id='spinner_wave'>\
                    <path class='spinnerWave' d='M159.2,199.6c72.4,22.7,117.7,163.3,167.6,163.3c92.3,0,80.1-237.2,177.5-237.2c97.4,0,96.9,237.2,194.3,237.2\
                        S796.1,49.7,894.1,44.1'/>\
                    <path class='spinnerWave' d='M161.2,194.3c78.9,27.6,115.3,178.6,164.5,181.4c90.3,4.7,81.1-265.6,178.5-265.6\
                        c97.4,0,96.9,265.6,194.3,265.6S796.3,58.6,894.3,53.1'/>\
                    <path class='spinnerWave' d='M163.3,189c85.4,32.5,112.8,194.1,161.5,199.4c88.2,9.6,82.1-294,179.5-294c97.4,0,97,294,194.4,294\
                        s97.8-321,195.8-326.3'/>\
                    <path class='spinnerWave' d='M165.3,183.7c91.9,37.3,110.3,209.5,158.4,217.5c86.2,14.3,83.2-322.4,180.6-322.4\
                        c97.4,0,97,322.4,194.4,322.4s98-324.9,196-330'/>\
                    <path class='spinnerWave' d='M167.3,178.5c98.4,42.2,107.8,224.8,155.4,235.5c84.1,19,84.2-350.8,181.6-350.8\
                        c97.4,0,97.1,350.8,194.5,350.8S796.9,85.1,894.9,80.3'/>\
                    <path class='spinnerWave' d='M169.4,173.2c104.9,47.1,105.4,240.1,152.4,253.6c82.1,23.6,85.2-379.2,182.6-379.2\
                        c97.4,0,97.1,379.2,194.5,379.2S797.1,94,895.1,89.3'/>\
                    <path class='spinnerWave' d='M171.4,167.9c111.4,51.9,102.9,255.3,149.3,271.6C400.8,467.7,406.9,32,504.3,32\
                        c97.4,0,97.1,407.6,194.5,407.6s98.5-336.7,196.5-341.2'/>\
                    <path class='spinnerWave' d='M173.4,162.6c117.9,56.8,100.5,270.6,146.3,289.7c78.1,32.7,87.2-436,184.6-436c97.4,0,97.2,436,194.6,436\
                        s98.7-340.7,196.7-344.9'/>\
                </g>\
            </svg>\
            </div>\
            <div class='spinner_text'>swizzling your song...</div>\
        </div>\
        ");
    } else {
        $("#msg").fadeIn().html("\
                        <span style='background-color: rgba(255, 0, 0, 0.6);\
                                     color: #fff;\
                                     font-size: 16px;\
                                     padding: 10px 20px;\
                                     margin: 10px;\
                                     border: 1px solid #f00;\
                                     border-radius: 5px;'>\
                            <span style='margin-right:15px;'>&#129302;</span>\
                            You didn\'t upload a file. Nothing to swizzle here :(\
                        </span>").delay(2000).fadeOut();
        event.preventDefault();        
    }
});