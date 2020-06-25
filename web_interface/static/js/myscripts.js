$(document).ready(function(){

    $("img.img-responsive.upload-img").click(function(){
        var path = $(this).attr('src').split('/')
        var filename = path[path.length - 1];
        $.ajax({
            url: "/",
            type: "get",
            data: {'filename': filename},
            success: function(response) {
                jQuery("#loadingModal").modal("show");
                $("#loadingModal").modal("show");

                window.location = "/";
                console.log('Success!');
                console.log(response);
        	},
        	error: function(error){
        	    console.log('Error!');
        	    console.log(error);
        	}
        });
    });

    $(".thumb").click(function(){
        var index = $(this).attr('data-slide-to');
        $(".top-pred").remove();
        if (index != 0) {
            var classes = $(this).attr('class_names');
            var scores = $(this).attr('scores');
            classes = classes.replace("[","").replace("]","").split(',');
            scores = scores.replace("[","").replace("]","").split(',');
            $(".top-pred-col").append(`<div class='top-pred row'></div>`);
            $(".top-pred").append(`<h2>Top five predictions</h2>`);
            $(".top-pred").append(`<div class='top-pred-list list-group'></div>`);
            for (i = 0; i < classes.length; i++) {
                var class_name = classes[i].replace("'", "");
                var score = scores[i].replace("'", "");
                $(".top-pred-list").append(`<li label-id> ${class_name} ${score}%</li>`);
            }
        }
    });

    $(".dropdown-menu li a").click(function(){
        var modelName = $(this).text();
        jQuery("#loadingModal").modal("show");
        var title = 'Loading classification model...';
        jQuery('#loadingModal').find('.modal-title').text(title);
        $.ajax({
            url: "/",
            type: "get",
            data: {'model_name': modelName},
            success: function(response) {
                window.location = "/";
                console.log('Success!');
                console.log(response);
        	},
        	error: function(error){
        	    console.log('Error!');
        	    console.log(error);
        	}
        });
    });
});